import os, logging, httpx, subprocess
from pathlib import Path
from httpx import Response, HTTPStatusError, RequestError

from llamaserve.settings import Settings
from llamaserve.utils import Utils


class LlamaServe:
    """Serves llama models locally"""

    def __init__(self) -> None:
        self.__logger: logging.Logger = logging.getLogger()
        self.__config: Settings = Settings()

    def unpack(self) -> bool:
        """
        Download and extract model weights from S3

        Returns:
            bool: whether both operations were successful
        """
        return self._get_weights(self.__config.WEIGHTS.KEY) and self.__unzip_weights()

    def serve(self) -> None:
        """Serve the model via vLLM"""
        proc = subprocess.Popen(
            [
                'vllm',
                'serve',
                Path(self._get_weights_path()).parent,
                '--port',
                str(self.__config.SERVER.PORT),
                '--dtype',
                self.__config.SERVER.PRECISION,
                '--max-model-len',
                str(self.__config.SERVER.MAX_MODEL_LENGTH),
            ]
        )
        try:
            proc.wait()
        except KeyboardInterrupt:
            print('\nShutting down...')
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print('Force killing...')
                proc.kill()
            print('Server stopped.')

    def _get_weights_path(self) -> str:
        return f'assets/weights/{self.__config.WEIGHTS.PATH}'

    def _get_weights_url(self) -> str:
        return f'https://{self.__config.WEIGHTS.ID}.execute-api.{self.__config.WEIGHTS.AWS_REGION}.amazonaws.com/dist/{self.__config.WEIGHTS.PATH}'

    def _get_weights(self, key: str) -> bool:
        if not os.path.isfile(self._get_weights_path()):
            try:
                response: Response = httpx.get(
                    self._get_weights_url(),
                    headers={'x-api-key': key},
                )
                response.raise_for_status()
            except HTTPStatusError as e:
                self.__logger.error(
                    f'Unable to download weights (details: HTTP error {e.response.status_code}: {e.response.text})'
                )
                return False
            except RequestError as e:
                self.__logger.error(
                    f'Unable to download weights (details: Request failed: {e})'
                )
                return False
            os.makedirs(os.path.dirname(self._get_weights_path()), exist_ok=True)
            open(self._get_weights_path(), 'wb').write(response.content)
        return True

    def __unzip_weights(self) -> bool:
        if Utils.one_file(self._get_weights_path()):
            try:
                Utils.unzip(self._get_weights_path())
            except Exception as e:
                self.__logger.error(f'Unable to extract weights (details: {e})')
                return False
        return True
