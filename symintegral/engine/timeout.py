"""
Wall-clock limits for single integration steps.

`run_with_timeout` uses cysignals alarms once Sage is loaded (they can
interrupt Maxima/FriCAS), and ``setitimer`` otherwise.
"""

import signal
import time
from typing import Any, Callable, Optional

from .. import sage as sage_mod


class StepTimeout(Exception):
    """An integration step ran out of time."""


def _raise_timeout(signum: int,
                   frame: Any) -> None:
    """
    SIGALRM handler of `run_with_timeout` without Sage.

    Parameters
    ----------
    signum : int
        Signal number.
    frame : frame
        Interrupted stack frame.

    Raises
    ------
    StepTimeout
        Always.
    """
    raise StepTimeout()


def _run_with_sage_alarm(fn: Callable[[], Any],
                         seconds: float) -> Any:
    """
    Run `fn` under a cysignals alarm (Sage loaded).

    Once Sage is loaded it owns SIGALRM, and only its alarm interrupts
    Maxima and FriCAS.

    Parameters
    ----------
    fn : callable
        Function without arguments.
    seconds : float
        Time limit.

    Returns
    -------
    Any
        Result of `fn`.

    Raises
    ------
    StepTimeout
        If the alarm fires. Sage's pexpect interfaces (FriCAS) turn the
        alarm into a KeyboardInterrupt; one arriving clearly before the
        deadline is a genuine Ctrl-C and re-raised.
    """
    from cysignals.alarm import AlarmInterrupt, alarm, cancel_alarm
    start = time.monotonic()
    alarm(seconds)
    try:
        return fn()
    except AlarmInterrupt:
        raise StepTimeout() from None
    except KeyboardInterrupt:
        if time.monotonic() - start < 0.99 * seconds:
            raise
        raise StepTimeout() from None
    finally:
        cancel_alarm()


def _run_with_itimer(fn: Callable[[], Any],
                     seconds: float) -> Any:
    """
    Run `fn` under ``signal.setitimer`` (Sage not loaded).

    Parameters
    ----------
    fn : callable
        Function without arguments.
    seconds : float
        Time limit.

    Returns
    -------
    Any
        Result of `fn`.

    Raises
    ------
    StepTimeout
        If the timer fires. The previous SIGALRM handler is restored,
        unless `fn` imported Sage: cysignals then owns SIGALRM, and
        restoring the old (default) handler would make later Sage alarms
        terminate the process.
    """
    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    start = time.monotonic()
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn()
    except KeyboardInterrupt:
        # Sage imported inside fn: its handler turns the timer into an
        # AlarmInterrupt (a KeyboardInterrupt); a genuine Ctrl-C arrives
        # clearly before the deadline and is re-raised
        if time.monotonic() - start < 0.99 * seconds:
            raise
        raise StepTimeout() from None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if not sage_mod.sage_loaded():
            signal.signal(signal.SIGALRM, previous)


def run_with_timeout(fn: Callable[[], Any],
                     seconds: Optional[float]) -> Any:
    """
    Run a function with a wall-clock limit.

    Only in the main thread, on Unix. Uses cysignals' alarm once Sage is
    loaded, else ``signal.setitimer``.

    Parameters
    ----------
    fn : callable
        Function without arguments.
    seconds : float or None
        Time limit; None or 0 runs `fn` without a limit.

    Returns
    -------
    Any
        Result of `fn`.

    Raises
    ------
    StepTimeout
        If `fn` takes longer than `seconds`.
    """
    if not seconds:
        return fn()
    if sage_mod.sage_loaded():
        return _run_with_sage_alarm(fn, seconds)
    return _run_with_itimer(fn, seconds)
