from llamaserve.llamaserve import LlamaServe


class TestLlamaServe(LlamaServe):

    __test__ = False

    def get_weights_path(self) -> str:
        return self._get_weights_path()

    def get_presigned_generation_url(self) -> str:
        return self._get_presigned_generation_url()

    def get_weights(self, key: str) -> bool:
        return self._get_weights(key)
