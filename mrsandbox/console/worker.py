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
import multiprocessing
import os
import pathlib

#************#
#   worker   #
#************#
# the page's process holds the pool here, and the worker process its one session, kept with the
# graphs it has built from one task to the next
WORKER = {}


#*************#
#   started   #
#*************#
def started():
    """The console's worker: one spawned process, `working` five steps below the page, warmed at
    once on the default room's card (`worked`)."""
    pool = concurrent.futures.ProcessPoolExecutor(1, multiprocessing.get_context("spawn"),
                                                  initializer=working, initargs=(5,))
    pool.submit(worked, None, "run")
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


#************#
#   worked   #
#************#
def worked(state, task, *args):
    """In the worker process: the Session method `task` on `args` with the page's `state`
    (Session.remote), returning its result, the clock and the timings. With no state the default
    room's card plays once and is thrown away, the console's warm-up at launch: its graphs are
    built while the operator plans."""
    from mrsandbox.console.session import Session

    session = WORKER.setdefault("session", Session())
    if state is None:
        session.run(session.plan(queued=False))
        return None
    vars(session).update(state)
    result = getattr(session, task)(*args)
    if task == "run" and result.data is not None:
        result.data = result.data.image.nibImage
    return result, session.clock_s, session._took
