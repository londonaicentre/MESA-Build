from llamaserve.llamaserve import LlamaServe


def run() -> None:
    llamaServe: LlamaServe = LlamaServe()
    llamaServe.unpack()
    llamaServe.serve()
