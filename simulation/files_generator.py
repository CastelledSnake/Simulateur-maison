from typing import List, Set
from random import Random

from model.storage.abstract_storage_tier import Storage
from model.storage.hdd_storage_tier import HDD
from model.storage.ssd_storage_tier import SSD
from model.storage.file import File


def files_generator(hdds: List[HDD], ssds: List[SSD], number_of_files: int, min_file_size: int, max_file_size: int,
                    rng: Random) -> Set[File]:
    """
    Creates a set of Files that can be implemented in a simulation.
    :param hdds:
    :param ssds:
    :param number_of_files:
    :param min_file_size:
    :param max_file_size:
    :param rng: A Random instance.
    :return:
    """
    files = set()
    for file_count in range(number_of_files):
        files.add(File(name=f'file_{file_count}.txt',
                       size=rng.randint(min_file_size, max_file_size),
                       preferred_storage=rng.choice(hdds + ssds)))
    return files_allocation(files, hdds, ssds, rng)


def files_allocation(files: Set[File], hdds: List[Storage], ssds: List[Storage], rng: Random) -> Set[File]:
    """
    Allocates a set of newly-generated Files to Storage devices given in a list.
    :param files:
    :param hdds:
    :param ssds:
    :param rng:
    :return:
    """
    for file in files:
        # If there is enough space available on the preferred storage, the file is allocated to it.
        if file.preferred_storage.capacity - file.preferred_storage.occupation >= file.size:
            assert file.storage != file.preferred_storage
            assert file not in file.preferred_storage.files
            file.storage = file.preferred_storage
            file.preferred_storage.files.add(file)
            file.preferred_storage.occupation += file.size
        else:
            if file.preferred_storage in hdds:
                available_media = [media for media in hdds if media.capacity - media.occupation >= file.size]
            else:
                available_media = [media for media in ssds if media.capacity - media.occupation >= file.size]
            if len(available_media) == 0:
                available_media = [media for media in ssds + hdds if
                                   media.capacity - media.occupation >= file.size]
            if len(available_media) == 0:
                raise RuntimeError(f'The following file could not fit in any storage media: {file}')
            media = rng.choice(available_media)
            assert file.storage != file.preferred_storage
            assert file not in file.preferred_storage.files
            file.storage = file.preferred_storage
            media.files.add(file)
            media.occupation += file.size
    return files
