import zipfile, tarfile, shutil, pathlib


class Utils:

    @staticmethod
    def unzip(path: str) -> bool:
        source: pathlib.Path = pathlib.Path(path)
        target: pathlib.Path = source.parent
        if not source.exists():
            raise FileNotFoundError(f'zip file not found: {source}')
        with zipfile.ZipFile(source) as zip:
            if zip.testzip() is not None:
                raise zipfile.BadZipFile('corrupted zip file')
            total_size: int = sum(info.file_size for info in zip.infolist())
            free_space: int = shutil.disk_usage(target).free
            if total_size > free_space:
                raise OSError(
                    f'insufficient disk space: {free_space} available but need {total_size}'
                )
            zip.extractall(target)
        return True

    @staticmethod
    def untar(path: str) -> bool:
        source: pathlib.Path = pathlib.Path(path)
        target: pathlib.Path = source.parent
        if not source.exists():
            raise FileNotFoundError(f'tar file not found: {source}')
        with tarfile.open(source, 'r:*') as tar:
            try:
                members: list[tarfile.TarInfo] = tar.getmembers()
            except tarfile.TarError:
                raise tarfile.TarError('corrupted tar file')
            total_size: int = sum(member.size for member in members if member.isfile())
            free_space: int = shutil.disk_usage(target).free
            if total_size > free_space:
                raise OSError(
                    f'insufficient disk space: {free_space} available but need {total_size}'
                )
            tar.extractall(target)
        return True

    @staticmethod
    def one_file(path: str) -> bool:
        file: pathlib.Path = pathlib.Path(path)
        if not file.is_file():
            raise ValueError(f'{file} is not a file.')
        files = [item for item in file.parent.iterdir() if item.is_file()]
        return len(files) == 1 and files[0] == file
