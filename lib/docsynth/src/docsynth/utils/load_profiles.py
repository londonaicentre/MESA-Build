import random

from schemallama_types.assets import SchemaLlamaAssets

"""
load_profiles.py - loads in cancer & molecular profiles
"""


class ProfileLoader:
    def __init__(self, assets: SchemaLlamaAssets):
        self.__assets = assets
        self.all_profiles = []
        self.profile_files = []

    def load_all_profiles(self):
        self.all_profiles.extend(self.__assets.load_all_profiles())
        return self.all_profiles

    def load_profiles_from_files(self, filenames):
        self.all_profiles.extend(self.__assets.load_profiles_from_files(filenames))
        return self.all_profiles

    def get_random_profile(self):
        if not self.all_profiles:
            raise ValueError("No profiles loaded.")
        return random.choice(self.all_profiles)

    def get_sequential_profiles(self):
        if not self.all_profiles:
            raise ValueError("No profiles loaded.")
        for profile in self.all_profiles:
            yield profile

    def format_profile_prompt(self, profile):
        return self.__assets.format_profile_prompt(profile)

    def get_profile_count(self):
        return self.__assets.get_profile_count()
