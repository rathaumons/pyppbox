# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Own GUI subprocess groups without importing inference backends."""
from contextlib import contextmanager
import os
import signal
import subprocess


def watch_gui_parent():
    """Stop a POSIX GUI session if its explicitly recorded parent disappears.

    Called only by GUI child entry points. Windows jobs enforce this in the OS.
    Requiring this process to lead its own session prevents signaling a host's
    terminal/process group when a helper is imported or called directly.
    """
    expected=os.environ.get('PYPPBOX_GUI_PARENT_PID','')
    if os.name=='nt' or not expected.isdecimal() or int(expected)<=0 or os.getsid(0)!=os.getpid():
        return
    import threading
    import time
    def monitor():
        while os.getppid()==int(expected): time.sleep(.05)
        os.killpg(os.getpid(),signal.SIGKILL)
    threading.Thread(target=monitor,name='pyppbox-gui-owner',daemon=True).start()


class _WindowsJob:
    """A private kill-on-close job; children start suspended until assigned."""
    def __init__(self):
        import ctypes as c
        from ctypes import wintypes as w
        self.c=c
        self.api=c.WinDLL('kernel32',use_last_error=True)
        class Basic(c.Structure):
            _fields_=[('ProcessTime',c.c_longlong),('JobTime',c.c_longlong),('Flags',w.DWORD),
                ('MinWorkingSet',c.c_size_t),('MaxWorkingSet',c.c_size_t),('ActiveProcesses',w.DWORD),
                ('Affinity',c.c_size_t),('Priority',w.DWORD),('Scheduling',w.DWORD)]
        class IO(c.Structure):
            _fields_=[(name,c.c_ulonglong) for name in ('ReadOps','WriteOps','OtherOps','ReadBytes','WriteBytes','OtherBytes')]
        class Extended(c.Structure):
            _fields_=[('Basic',Basic),('IO',IO),('ProcessMemory',c.c_size_t),('JobMemory',c.c_size_t),
                ('PeakProcessMemory',c.c_size_t),('PeakJobMemory',c.c_size_t)]
        class Thread(c.Structure):
            _fields_=[('size',w.DWORD),('usage',w.DWORD),('id',w.DWORD),('pid',w.DWORD),
                      ('priority',w.LONG),('delta',w.LONG),('flags',w.DWORD)]
        self.Thread=Thread
        signatures={
            'CreateJobObjectW':([c.c_void_p,w.LPCWSTR],w.HANDLE),
            'SetInformationJobObject':([w.HANDLE,c.c_int,c.c_void_p,w.DWORD],w.BOOL),
            'AssignProcessToJobObject':([w.HANDLE,w.HANDLE],w.BOOL),
            'OpenProcess':([w.DWORD,w.BOOL,w.DWORD],w.HANDLE),
            'CreateToolhelp32Snapshot':([w.DWORD,w.DWORD],w.HANDLE),
            'Thread32First':([w.HANDLE,c.POINTER(Thread)],w.BOOL),
            'Thread32Next':([w.HANDLE,c.POINTER(Thread)],w.BOOL),
            'OpenThread':([w.DWORD,w.BOOL,w.DWORD],w.HANDLE),
            'ResumeThread':([w.HANDLE],w.DWORD),
            'CloseHandle':([w.HANDLE],w.BOOL),
        }
        for name,(args,result) in signatures.items():
            function=getattr(self.api,name); function.argtypes=args; function.restype=result
        self.handle=self.api.CreateJobObjectW(None,None)
        if not self.handle: raise c.WinError(c.get_last_error())
        limits=Extended(); limits.Basic.Flags=0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle,9,c.byref(limits),c.sizeof(limits)):
            error=c.WinError(c.get_last_error()); self.close(); raise error

    def attach_and_resume(self,process):
        c=self.c
        handle=self.api.OpenProcess(0x100|0x1,False,process.pid)  # SET_QUOTA | TERMINATE
        if not handle: raise c.WinError(c.get_last_error())
        try:
            if not self.api.AssignProcessToJobObject(self.handle,handle):
                raise c.WinError(c.get_last_error())
        finally: self.api.CloseHandle(handle)
        snapshot=self.api.CreateToolhelp32Snapshot(0x4,0)  # TH32CS_SNAPTHREAD
        if snapshot==c.c_void_p(-1).value: raise c.WinError(c.get_last_error())
        try:
            entry=self.Thread(); entry.size=c.sizeof(entry)
            found=self.api.Thread32First(snapshot,c.byref(entry))
            while found:
                if entry.pid==process.pid:
                    thread=self.api.OpenThread(0x2,False,entry.id)  # THREAD_SUSPEND_RESUME
                    if not thread: raise c.WinError(c.get_last_error())
                    try:
                        if self.api.ResumeThread(thread)==0xffffffff: raise c.WinError(c.get_last_error())
                    finally: self.api.CloseHandle(thread)
                    return
                found=self.api.Thread32Next(snapshot,c.byref(entry))
            raise OSError('Cannot find the suspended GUI child thread.')
        finally: self.api.CloseHandle(snapshot)

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle); self.handle=None


@contextmanager
def gui_process(command, *, popen=None, env=None, **options):
    """Reap the GUI child and stop its owned descendants on completion/interruption.

    Windows uses a nested, private job. POSIX children own separate sessions and
    GUI entry points monitor parent loss, including abrupt launcher termination.
    Config paths and the caller's environment are not modified.
    """
    popen=subprocess.Popen if popen is None else popen
    child_env=dict(os.environ if env is None else env)
    job=_WindowsJob() if os.name=='nt' else None
    owns_group=False
    process=None
    if job is not None:
        options['creationflags']=options.get('creationflags',0)|0x4  # CREATE_SUSPENDED
    else:
        owns_group=True
        options['start_new_session']=True
        child_env['PYPPBOX_GUI_PARENT_PID']=str(os.getpid())
    try:
        process=popen(command,env=child_env,**options)
        if job is not None: job.attach_and_resume(process)
        yield process
    finally:
        if job is not None:
            job.close()
        elif process is not None and owns_group:
            try: os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError: pass
        elif process is not None and process.poll() is None:
            process.terminate()
        if process is not None:
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)
            finally:
                if owns_group:
                    try: os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError: pass
