####################################################################################################
#                                             worker.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 30/10/26                                                                                #
#                                                                                                  #
# Purpose: The console's worker: the process its simulations run in, so the page's process never   #
#          traces nor compiles. Light on purpose: the page's process imports it before the engine. #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import concurrent.futures
import dataclasses
import multiprocessing
import os
import pathlib
import time

#************#
#   worker   #
#************#
# the page's process holds the pool here, and the worker process its one session, kept with the
# graphs it has built from one task to the next
WORKER = {}
# the library's families `mrsandbox-console --warm` builds (`warm`), the default card's first:
# the card's sequence and whether its pulses are played (HERCULES has no ideal design)
FAMILIES = (("PRESS", True), ("PRESS", False), ("STEAM", True), ("STEAM", False),
            ("sLASER", True), ("sLASER", False), ("MEGA-PRESS", True), ("MEGA-PRESS", False),
            ("HERMES GABA GSH", True), ("HERMES GABA GSH", False), ("HERCULES", True))


#*************#
#   started   #
#*************#
def started():
    """The console's worker: one spawned process, `working` five steps below the page, warmed at
    once on the default room's card (`worked`)."""
    pool = concurrent.futures.ProcessPoolExecutor(1, multiprocessing.get_context("spawn"),
                                                  initializer=working, initargs=(5,))
    pool.submit(worked, None, "warm")
    return pool


#*************#
#   working   #
#*************#
def working(nice=0):
    """Ready a process to simulate for the console: jax's compilation cache in the user's cache
    folder, so the engine's kernels compile once per machine and not once per launch, and the
    process `nice` steps below the page in the scheduler, so the page and the computer stay
    smooth while it works."""
    import keras

    os.nice(nice)
    if keras.backend.backend() == "jax":
        import jax

        jax.config.update("jax_compilation_cache_dir",
                          str(pathlib.Path.home() / ".cache" / "mrsandbox" / "jax"))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)


#**********#
#   warm   #
#**********#
def warm(session, families=FAMILIES, said=lambda text: None):
    """The designs' graphs of `families` built into this machine's cache, the default card's
    family first: each family's card (the session's card otherwise, four transients so an edited
    scheme has one each, no reference) is played once as a preview and thrown away, so its
    exports persist (Engine's, beside jax's cache); a family whose graphs this code built here
    before (a marker beside the exports,
    keyed by the engine's code salt) is skipped, so a launch after the first costs nothing.
    `said` hears each family's seconds. Returns [(sequence, played, seconds)] of the families
    built; nothing without a persistent cache (another backend)."""
    import jax

    from mrsandbox.engine.core import Engine

    directory = jax.config.jax_compilation_cache_dir
    if directory is None:
        return []
    marks = pathlib.Path(directory) / "warm"
    card, built = session.protocol, []
    try:
        for sequence, played in families:
            mark = marks / f"{sequence}, {'played' if played else 'ideal'}, {Engine._export_salt()}"
            if mark.exists():
                continue
            session.protocol = dataclasses.replace(card, sequence=sequence, real=played,
                                                   averages=4, water_reference=0)
            start = time.perf_counter()
            session.run(session.plan(queued=False))
            marks.mkdir(parents=True, exist_ok=True)
            mark.touch()
            built.append((sequence, played, time.perf_counter() - start))
            said(f"{sequence}, {'played' if played else 'ideal'}: {built[-1][2]:.0f} s")
    finally:
        session.protocol = card
    return built


#************#
#   worked   #
#************#
def worked(state, task, *args):
    """In the worker process: the Session method `task` on `args` with the page's `state`
    (Session.remote), returning its result, the clock and the timings. With no state the
    default card's family is warmed (`warm`), the console's warm-up at launch: its graphs are
    built while the operator plans; the other families compile on their first exam or by
    `mrsandbox-console --warm`."""
    from mrsandbox.console.session import Session

    session = WORKER.setdefault("session", Session())
    if state is None:
        warm(session, families=((session.protocol.sequence, session.protocol.real),))
        return None
    vars(session).update(state)
    result = getattr(session, task)(*args)
    if task == "run" and result.data is not None:
        result.data = result.data.image.nibImage
    return result, session.clock_s, session._took
