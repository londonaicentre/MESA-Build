from finetune.llama import FineTuner


class TestFineTuner(FineTuner):
    __test__ = False

    def generate_train_file(self, samples_input_folder: str) -> bool:
        return self._generate_train_file(samples_input_folder)
