"""Machine abstraction (local / remote via SSH+paramiko).

Provides the Machine class for command execution and filesystem queries,
and the Machines_handler class for SFTP-based file transfer.
"""

# Copyright (C) 2024- Kosuke Nakano
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import concurrent.futures
import contextlib
import os
import pathlib
import random
import re
import shlex
import shutil
import stat
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger

import paramiko
import yaml

from ._config import get_config_dir, template_dir

logger = getLogger("jqmc-workflow").getChild(__name__)


class Machine:
    """Represents a local or remote machine."""

    ssh_retry_time = 120
    ssh_retry_max_num = 10

    def __init__(self, machine: str):
        cfg = get_config_dir()
        self.machine_info_yaml = os.path.join(cfg, "machine_data.yaml")

        # Bootstrap config dir on first run
        if not os.path.isdir(cfg):
            logger.info(f"{cfg} not found. Bootstrapping from template (first run).")
            shutil.copytree(template_dir, cfg)
            logger.info(f"{cfg} has been generated.")
            logger.info(f"Please edit {self.machine_info_yaml}")
            raise FileNotFoundError(f"Please configure {self.machine_info_yaml} before running.")

        # Load machine data
        try:
            with open(self.machine_info_yaml) as yf:
                self.data = yaml.safe_load(yf)[machine]
        except FileNotFoundError:
            logger.error(f"Config file {self.machine_info_yaml} not found!")
            raise
        except KeyError:
            logger.error(f"machine='{machine}' is not defined in {self.machine_info_yaml}. Please add it.")
            raise

        self.__name = machine
        self.ssh_status = False
        self._proxy_cmd = None  # track ProxyCommand for cleanup
        # Serializes access to the shared SFTPClient. paramiko's SFTPClient
        # multiplexes request ids on a single channel and is not
        # documented as thread-safe; concurrent put/get/listdir from
        # different threads can racemap responses. The RLock allows
        # recursive directory walks (``_get_sftp_dir`` / ``_put_sftp_dir``)
        # to hold the lock for the duration of the walk without deadlock.
        self._sftp_lock = threading.RLock()

        # Validate ssh_host for remote machines
        if self.machine_type == "remote" and "ssh_host" not in self.data:
            raise KeyError(
                f"machine='{machine}' has machine_type='remote' but no 'ssh_host' field "
                f"in {self.machine_info_yaml}. "
                f"Please add 'ssh_host' (e.g. the Host alias in ~/.ssh/config)."
            )

    def __del__(self):
        """Safety net: clean up SSH resources if not explicitly closed."""
        try:
            if getattr(self, "ssh_status", False):
                self.ssh_close()
        except Exception:
            pass

    # -- SSH management ----------------------------------------------

    @staticmethod
    def _kill_proxy_process(proxy_cmd):
        """Force-kill a ProxyCommand subprocess and close its pipe FDs.

        paramiko's ProxyCommand.close() only sends SIGTERM without
        wait() or closing pipes, leaving zombie processes and leaked
        file descriptors.  This method ensures full cleanup.

        After cleanup, the ProxyCommand's close() is neutralised so
        that paramiko's transport thread does not attempt to kill the
        already-reaped process (which would log a ProcessLookupError).
        """
        if proxy_cmd is None:
            return
        proc = getattr(proxy_cmd, "process", None)
        if proc is None:
            return
        # Close pipe file descriptors first
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            if pipe is not None:
                try:
                    pipe.close()
                except Exception:
                    pass
        # Kill the process (SIGKILL -- SIGTERM may be ignored)
        try:
            proc.kill()
        except OSError:
            pass  # already dead
        # Reap the zombie so the PID is released
        try:
            proc.wait(timeout=3)
        except Exception:
            pass
        # Neutralise the ProxyCommand so that paramiko's transport
        # thread does not try os.kill() on the now-dead PID.
        proxy_cmd.close = lambda: None

    def ssh_open(self):
        if self.machine_type != "remote":
            return
        if self.ssh_status:
            logger.debug(f"SSH already open (id={id(self.ssh)})")
            return

        # Note: this is a *synchronous* sleep, even when ``ssh_open`` is
        # reached from an asyncio coroutine (e.g. ``_submit_and_wait``).
        # It blocks the event loop for 3-6 s.  Parallel workflows
        # (Launcher / LRDMC_Ext) effectively serialise here.  Replacing
        # this whole layer with asyncssh + ``await asyncio.sleep`` would
        # restore parallelism; until then, prefer fewer SSH opens per
        # workflow rather than tighter polling intervals.
        rw = random.randint(3, 6)
        logger.info(f"  Wait {rw}s before opening SSH to {self.name}")
        time.sleep(rw)

        ssh_config = paramiko.SSHConfig()
        config_file = os.path.join(os.getenv("HOME"), ".ssh/config")
        try:
            with open(config_file) as fh:
                ssh_config.parse(fh)
        except FileNotFoundError:
            logger.error(f"SSH config file ({config_file}) is required.")
            raise

        # Workaround: paramiko raises KeyError('canonicaldomains') when
        # CanonicalizeHostname is set but CanonicalDomains is missing.
        try:
            lkup = ssh_config.lookup(self.ssh_host)
        except KeyError:
            from io import StringIO

            with open(config_file) as fh:
                original = fh.read()
            # Disable CanonicalizeHostname to avoid the paramiko bug
            patched = re.sub(
                r"(?im)^(\s*CanonicalizeHostname\s+)yes",
                r"\1no",
                original,
            )
            ssh_config = paramiko.SSHConfig()
            ssh_config.parse(StringIO(patched))
            lkup = ssh_config.lookup(self.ssh_host)

        hostname = lkup["hostname"]
        username = lkup["user"]
        key_filename = lkup["identityfile"]

        proxy_command = lkup.get("proxycommand")
        proxy_flag = proxy_command is not None

        self.username = username
        self.ssh = paramiko.SSHClient()
        self.ssh.load_system_host_keys()

        for tt in range(self.ssh_retry_max_num):
            proxy_cmd = None
            try:
                kwargs = dict(
                    hostname=hostname,
                    username=username,
                    key_filename=key_filename,
                )
                if proxy_flag:
                    proxy_cmd = paramiko.ProxyCommand(proxy_command)
                    kwargs["sock"] = proxy_cmd
                self.ssh.connect(**kwargs)
                # Enable SSH keepalive so the client detects dead connections
                # early instead of discovering them at close() time.
                self.ssh.get_transport().set_keepalive(30)
                self._proxy_cmd = proxy_cmd  # save reference for cleanup
                logger.info(f"  SSH connected (attempt {tt + 1})")
                break
            except paramiko.ssh_exception.SSHException:
                # Clean up the ProxyCommand from this failed attempt
                self._kill_proxy_process(proxy_cmd)
                if tt == self.ssh_retry_max_num - 1:
                    # Re-raise immediately on the final attempt; no point
                    # sleeping ssh_retry_time only to give up afterwards.
                    logger.error("SSH connect failed after all retries.")
                    raise
                logger.warning(f"SSH connect failed (attempt {tt + 1}). Retrying in {self.ssh_retry_time}s.")
                time.sleep(self.ssh_retry_time)

        self.sftp = self.ssh.open_sftp()
        self.ssh_status = True

    def ssh_close(self):
        if self.machine_type != "remote" or not self.ssh_status:
            return

        # Save proxy reference before closing -- ssh.close() may clear it
        proxy_cmd = self._proxy_cmd
        self._proxy_cmd = None

        timeout_sec = 5.0
        for obj_name, obj in [("sftp", self.sftp), ("ssh", self.ssh)]:
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(obj.close)
            try:
                future.result(timeout=timeout_sec)
                logger.debug(f"{obj_name}.close() ok")
            except Exception as e:
                logger.warning(f"{obj_name}.close() failed ({e.__class__.__name__}: {e}) -- abandoning")
                future.cancel()
            finally:
                executor.shutdown(wait=False)

        # Force-kill the ProxyCommand subprocess and close its pipe FDs.
        # This is essential because:
        #  - paramiko's ProxyCommand.close() only sends SIGTERM, never
        #    calls wait() or closes the subprocess pipes.
        #  - If ssh.close() timed out above, the ProxyCommand process
        #    is still alive with open file descriptors.
        self._kill_proxy_process(proxy_cmd)

        del self.ssh
        del self.sftp
        self.ssh_status = False

    @contextlib.contextmanager
    def sftp_session(self):
        """Yield the shared :class:`paramiko.SFTPClient` under a thread lock.

        All SFTP access in this module MUST go through this context
        manager. paramiko's SFTPClient is not officially thread-safe
        and concurrent put/get/listdir from sibling coroutines (which
        ``Launcher`` runs in their own ThreadPoolExecutor workers via
        ``_run_remote`` / ``_sftp_lstat_with_retry``) can race the
        internal request-id demux. The lock is an ``RLock`` so that
        recursive directory walks (``Machines_handler._get_sftp_dir``)
        can hold it for the duration of the walk without deadlocking
        on themselves.

        Note: this returns the same SFTPClient instance every call --
        we do *not* re-open the SFTP channel per operation, since the
        Transport is shared via the Machine registry and re-opening
        would otherwise multiply channel open/close on every file
        transfer.
        """
        self.ssh_open()
        with self._sftp_lock:
            yield self.sftp

    # -- Properties (read from machine_data.yaml) ------------------

    _MISSING = object()  # sentinel for _get() default detection

    def _get(self, key, default=_MISSING):
        try:
            return self.data[key]
        except KeyError:
            if default is not self._MISSING:
                return default
            raise KeyError(f"'{key}' not found for machine '{self.name}'")

    @property
    def name(self):
        return self.__name

    @property
    def ssh_host(self) -> str:
        """SSH host identifier (Host alias in ~/.ssh/config or user@hostname)."""
        return self._get("ssh_host")

    @property
    def machine_type(self) -> str:
        val = self._get("machine_type")
        if val not in {"local", "remote"}:
            raise ValueError(f"machine_type must be 'local' or 'remote', got '{val}'")
        return val

    @property
    def username(self):
        try:
            return self._username
        except AttributeError:
            return os.getlogin()

    @username.setter
    def username(self, value):
        self._username = value

    @property
    def ip(self):
        return self._get("ip", default=None)

    @property
    def workspace_root(self):
        return self._get("workspace_root", default=None)

    @property
    def queuing(self) -> bool:
        return self._get("queuing")

    @property
    def jobsubmit(self) -> str:
        return self._get("jobsubmit")

    @property
    def jobcheck(self) -> str:
        return self._get("jobcheck")

    @property
    def jobdel(self) -> str:
        return self._get("jobdel")

    @property
    def jobnum_index(self) -> int:
        return self._get("jobnum_index")

    # -- Command execution -----------------------------------------

    def run_command(self, command: str, execute_dir: str = None):
        if execute_dir:
            if self.machine_type == "remote":
                with self.sftp_session() as sftp:
                    fileattr = sftp.lstat(execute_dir)
                if not stat.S_ISDIR(fileattr.st_mode):
                    raise FileNotFoundError(f"{execute_dir} not found on remote machine.")
            else:
                if not os.path.isdir(execute_dir):
                    raise FileNotFoundError(f"{execute_dir} not found locally.")
            command_r = f"cd {shlex.quote(execute_dir)} && {command}"
        else:
            command_r = command

        if self.machine_type == "local":
            return self._run_local(command_r)
        return self._run_remote(command_r)

    def _run_local(self, command_r: str, max_retries: int = 3):
        """Run a shell command on the local machine.

        Only a timeout is retried (``max_retries`` attempts, 60 s apart): it
        may be a transient load problem. A non-zero exit status of a local
        command is deterministic -- no network is involved -- so it is
        reported once, with the captured output, and raised immediately
        instead of being repeated ``max_retries`` times with long sleeps.
        """
        for attempt in range(max_retries):
            try:
                proc = subprocess.run(
                    command_r,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=1200,
                )
            except subprocess.TimeoutExpired:
                logger.warning(f"Local command timeout (attempt {attempt + 1}/{max_retries}): {command_r}")
                if attempt < max_retries - 1:
                    time.sleep(60)
                continue
            if proc.returncode == 0:
                return proc.stdout, proc.stderr
            stdout_tail = proc.stdout.strip()[-500:]
            stderr_tail = proc.stderr.strip()[-500:]
            logger.error(f"Local command returned rc={proc.returncode}: {command_r}")
            if stdout_tail:
                logger.error(f"  stdout: {stdout_tail}")
            if stderr_tail:
                logger.error(f"  stderr: {stderr_tail}")
            raise RuntimeError(f"Local command failed (rc={proc.returncode}): {command_r}\n{stderr_tail or stdout_tail}")

        raise RuntimeError(f"Local command timed out {max_retries} times: {command_r}")

    remote_command_timeout_sec = 1200  # match _run_local default

    def _run_remote(self, command_r: str):
        """Execute *command_r* on the remote host with a hard wall-time guard.

        ``recv_exit_status`` waits on a paramiko ``status_event`` that is
        *not* connected to socket-level keepalive, so a dead SSH session
        can hang the call indefinitely.  We run the entire exec/read
        sequence in a worker thread and enforce a timeout via
        :class:`concurrent.futures.Future`; on timeout the SSH session is
        torn down so the next call reconnects cleanly.
        """
        self.ssh_open()

        def _do_exec():
            try:
                pstdin, pstdout, pstderr = self.ssh.exec_command(command=command_r)
            except (paramiko.SSHException, OSError, EOFError):
                # Connection may have died (e.g. keepalive timeout during
                # a long asyncio.sleep between polls).  Reconnect once.
                logger.warning("SSH connection lost during exec_command; reconnecting...")
                self.ssh_close()
                self.ssh_open()
                pstdin, pstdout, pstderr = self.ssh.exec_command(command=command_r)
            try:
                exit_status = pstdout.channel.recv_exit_status()
                stdout = pstdout.read().decode("utf-8").strip()
                stderr = pstderr.read().decode("utf-8").strip()
            finally:
                for ch in (pstdin, pstdout, pstderr):
                    try:
                        ch.close()
                    except Exception:
                        pass
            return exit_status, stdout, stderr

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_do_exec)
        try:
            exit_status, stdout, stderr = future.result(timeout=self.remote_command_timeout_sec)
        except concurrent.futures.TimeoutError as exc:
            logger.error(f"Remote command timed out after {self.remote_command_timeout_sec}s: {command_r}")
            try:
                self.ssh_close()
            except Exception:
                pass
            raise RuntimeError(f"Remote command timed out after {self.remote_command_timeout_sec}s: {command_r}") from exc
        finally:
            executor.shutdown(wait=False)

        if exit_status != 0:
            logger.error(f"Remote command failed: {command_r}")
            logger.error(f"stdout={stdout}")
            logger.error(f"stderr={stderr}")
            raise RuntimeError(f"Remote command failed (exit={exit_status}): {command_r}")
        return stdout, stderr

    # -- Filesystem queries ----------------------------------------

    def _sftp_lstat_with_retry(self, path: str, max_retries=3, timeout_sec=5.0):
        self.ssh_open()

        def _locked_lstat():
            # Hold the SFTP lock for the duration of the lstat. We
            # deliberately use ``_sftp_lock`` (not ``sftp_session``) so
            # the lock is acquired in the worker thread; this keeps the
            # outer timeout enforced via ``future.result(timeout=...)``.
            with self._sftp_lock:
                return self.sftp.lstat(path)

        for attempt in range(1, max_retries + 1):
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(_locked_lstat)
            try:
                return future.result(timeout=timeout_sec)
            except Exception as e:
                logger.warning(f"SFTP lstat attempt {attempt}: {e}")
                if attempt == max_retries:
                    raise RuntimeError(f"SFTP lstat failed for '{path}'")
                time.sleep(1)
            finally:
                # wait=False: the lstat thread may be stuck on a dead connection;
                # waiting would hang forever.
                executor.shutdown(wait=False)

    def is_file(self, file_name: str) -> bool:
        if self.machine_type == "local":
            return os.path.isfile(file_name)
        try:
            fileattr = self._sftp_lstat_with_retry(file_name)
        except (RuntimeError, OSError):
            return False
        return stat.S_ISREG(fileattr.st_mode)

    def is_dir(self, dir_name: str) -> bool:
        if self.machine_type == "local":
            return os.path.isdir(dir_name)
        try:
            fileattr = self._sftp_lstat_with_retry(dir_name)
        except (RuntimeError, OSError):
            return False
        return stat.S_ISDIR(fileattr.st_mode)

    def exist(self, object_name: str) -> bool:
        if self.machine_type == "local":
            return os.path.exists(object_name)
        try:
            fileattr = self._sftp_lstat_with_retry(object_name)
        except (RuntimeError, OSError):
            return False
        return stat.S_ISDIR(fileattr.st_mode) or stat.S_ISREG(fileattr.st_mode)

    # -- Job list queries ------------------------------------------

    def get_job_list(self):
        return self.run_command(self.jobcheck)

    def get_job_list_as_text(self):
        stdout, _ = self.get_job_list()
        return stdout.split("\n")

    def delete_job(self, jobid):
        stdout, _ = self.run_command(f"{self.jobdel} {shlex.quote(str(jobid))}")
        return stdout.split("\n")


class Machines_handler:
    """Handles data transfer between localhost and a server machine.

    The client is always localhost -- only one Machine (server) is needed.
    """

    def __init__(self, machine: Machine):
        self.server_machine = machine

    def ssh_close(self):
        # No-op: the underlying Machine is shared via ``get_machine``
        # and torn down centrally by :py:func:`close_all_machines` at
        # workflow end. Tearing it down here would force the next
        # JobSubmission to re-authenticate (re-prompting OTP on
        # clusters that require it).
        return

    # -- put / get conveniences ------------------------------------

    def put(self, from_file, to_file, exclude_patterns=None):
        self._transfer(from_file, to_file, exclude_patterns, dir_transfer=False, direction="put")

    def put_dir(self, from_dir, to_dir, exclude_patterns=None):
        self._transfer(from_dir, to_dir, exclude_patterns, dir_transfer=True, direction="put")

    def get(self, from_file, to_file, exclude_patterns=None):
        self._transfer(from_file, to_file, exclude_patterns, dir_transfer=False, direction="get")

    def get_dir(self, from_dir, to_dir, exclude_patterns=None):
        self._transfer(from_dir, to_dir, exclude_patterns, dir_transfer=True, direction="get")

    # -- SFTP primitives -------------------------------------------

    def _get_sftp_file(self, source, target, exclude_patterns):
        if exclude_patterns and any(re.match(p, os.path.basename(source)) for p in exclude_patterns):
            return
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with self.server_machine.sftp_session() as sftp:
            sftp.get(source, target)

    def _put_sftp_file(self, source, target, exclude_patterns):
        if exclude_patterns and any(re.match(p, os.path.basename(source)) for p in exclude_patterns):
            return
        self.server_machine.run_command(f"mkdir -p {shlex.quote(os.path.dirname(target))}")
        with self.server_machine.sftp_session() as sftp:
            sftp.put(source, target)

    def _get_sftp_dir(self, source, target, exclude_patterns):
        os.makedirs(target, exist_ok=True)
        # Hold the SFTP lock for the entire recursive walk: the inner
        # listdir_attr / get pairs must not interleave with a sibling
        # transfer's requests. RLock makes the recursive ``_get_sftp_dir``
        # call below safe.
        with self.server_machine.sftp_session() as sftp:
            for item in sftp.listdir_attr(source):
                name = item.filename
                if exclude_patterns and any(re.match(p, name) for p in exclude_patterns):
                    continue
                remote_path = os.path.join(source, name)
                local_path = os.path.join(target, name)
                if stat.S_ISREG(item.st_mode):
                    sftp.get(remote_path, local_path)
                elif stat.S_ISDIR(item.st_mode):
                    self._get_sftp_dir(remote_path, local_path, exclude_patterns)

    def _put_sftp_dir(self, source, target, exclude_patterns):
        self.server_machine.run_command(f"mkdir -p {shlex.quote(target)}")
        with self.server_machine.sftp_session() as sftp:
            for item in os.listdir(source):
                if exclude_patterns and any(re.match(p, item) for p in exclude_patterns):
                    continue
                local_path = os.path.join(source, item)
                remote_path = os.path.join(target, item)
                if os.path.isfile(local_path):
                    sftp.put(local_path, remote_path)
                elif os.path.isdir(local_path):
                    self._put_sftp_dir(local_path, remote_path, exclude_patterns)

    # -- Core transfer logic ---------------------------------------

    def _transfer(self, from_path, to_path, exclude_patterns, dir_transfer, direction):
        exclude_patterns = exclude_patterns or []

        if not pathlib.Path(from_path).is_absolute():
            raise ValueError(f"from_path must be absolute: {from_path}")
        if not pathlib.Path(to_path).is_absolute():
            raise ValueError(f"to_path must be absolute: {to_path}")

        # Ensure target directory exists
        to_dir = os.path.dirname(to_path) if not dir_transfer else to_path
        if direction == "put":
            self.server_machine.run_command(f"mkdir -p {shlex.quote(to_dir)}")
        else:
            os.makedirs(to_dir, exist_ok=True)

        if self.server_machine.machine_type == "local":
            logger.debug("Server is localhost; no network transfer needed.")
            return

        logger.info(f"  Transfer ({direction}): {from_path} -> {to_path}")

        if direction == "put":
            if dir_transfer:
                self._put_sftp_dir(from_path, to_path, exclude_patterns)
            else:
                self._put_sftp_file(from_path, to_path, exclude_patterns)
        else:
            if dir_transfer:
                self._get_sftp_dir(from_path, to_path, exclude_patterns)
            else:
                self._get_sftp_file(from_path, to_path, exclude_patterns)


# -- Machine registry (one SSH transport per machine per process) -----
#
# Background. ``paramiko.SSHClient.connect()`` performs a full SSH
# authentication on every call -- it ignores OpenSSH's
# ``ControlMaster``/``ControlPath``/``ControlPersist`` settings because
# multiplexing lives in the OpenSSH client binary (UNIX socket), not in
# the SSH protocol. Clusters that require a one-time password / OTP
# therefore prompt on every ``connect()``.
#
# A fresh :class:`Machine` per :class:`JobSubmission` /
# :class:`Data_transfer`, torn down after each step, would re-authenticate
# (OTP) at every step. The registry below makes ``get_machine(name)`` return a
# process-wide singleton so all callers share one paramiko transport
# (and therefore one authentication). Concurrent channel opens
# (`exec_command`, SFTP) are paramiko-Transport-safe; the shared
# SFTPClient is additionally guarded by :pyattr:`Machine._sftp_lock`
# (see :py:meth:`Machine.sftp_session`).
#
# ``Machine.ssh_close()`` stays functional: it closes the SSH/SFTP
# but does NOT remove the entry from the registry. The
# ``_run_remote`` self-healing path
# (``ssh_close(); ssh_open()``) relies on this -- a dead transport can
# be torn down and reopened on the same instance without churning
# downstream references. To explicitly tear everything down (e.g. at
# the end of :py:meth:`Launcher.async_launch`), call
# :py:func:`close_all_machines`.
#
# Single-use entry points that want isolated short-lived connections
# (``probe_environment``, ``_cli.check_status``/``delete_job``) keep
# constructing ``Machine(name)`` directly; that bypasses the registry
# and the explicit ``ssh_close()`` they do in ``finally`` tears the
# isolated transport down cleanly.
# --------------------------------------------------------------------

_MACHINE_REGISTRY: dict[str, "Machine"] = {}
_REGISTRY_LOCK = threading.Lock()


def get_machine(name: str) -> "Machine":
    """Return the process-wide cached :class:`Machine` for *name*.

    Construction is cheap (no SSH dial-out); the first ``ssh_open()``
    on the returned instance is what actually authenticates. All
    subsequent callers asking for the same *name* receive the same
    instance, so the authenticated transport is reused -- one OTP
    prompt per process instead of one per workflow step.

    Thread-safe: a process-wide lock guards the dict during lookup
    and insertion. Per-instance state (SSH/SFTP open/close) uses the
    Machine's own locks once obtained.
    """
    with _REGISTRY_LOCK:
        m = _MACHINE_REGISTRY.get(name)
        if m is None:
            m = Machine(name)
            _MACHINE_REGISTRY[name] = m
        return m


def close_all_machines() -> None:
    """Close every cached Machine and clear the registry.

    Called once at the end of :py:meth:`Launcher.async_launch` to
    tear down the shared SSH transports for all machines touched
    during the run. Safe to call multiple times; safe to call when
    the registry is empty.
    """
    with _REGISTRY_LOCK:
        cached = list(_MACHINE_REGISTRY.items())
        _MACHINE_REGISTRY.clear()
    for name, m in cached:
        try:
            m.ssh_close()
        except Exception as e:
            logger.warning(f"close_all_machines: {name} ssh_close failed: {e}")


# -- Machine catalog (MCP adapter helpers) -------------------------


def list_machines() -> list[dict]:
    """Return a summary of all machines defined in ``machine_data.yaml``.

    Each entry contains the machine name and key configuration fields.
    Returns an empty list if the config file does not exist.
    """
    machine_data_path = os.path.join(get_config_dir(), "machine_data.yaml")
    if not os.path.isfile(machine_data_path):
        return []
    with open(machine_data_path) as f:
        data = yaml.safe_load(f)
    if not data:
        return []
    return [
        {
            "name": name,
            "machine_type": cfg.get("machine_type", "local"),
            "queuing": cfg.get("queuing", False),
            "ssh_host": cfg.get("ssh_host", ""),
            "workspace_root": cfg.get("workspace_root", ""),
        }
        for name, cfg in data.items()
    ]


def probe_environment(machine_name: str) -> dict:
    """Test connectivity to the named machine.

    For remote machines an SSH connection is attempted; for local machines
    reachability is always ``True``.  No software detection (jqmc, JAX, etc.)
    is performed -- that responsibility belongs to the MCP agent.
    """
    machine = Machine(machine_name)
    result: dict = {"machine_name": machine_name, "machine_type": machine.machine_type}
    try:
        if machine.machine_type == "remote":
            machine.ssh_open()
        result["reachable"] = True
    except Exception as e:
        result["reachable"] = False
        result["error"] = str(e)
    finally:
        if machine.machine_type == "remote" and machine.ssh_status:
            machine.ssh_close()
    return result
