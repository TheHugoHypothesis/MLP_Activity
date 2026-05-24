import numpy as np
from typing import List, Tuple


class DataLoader:
    @staticmethod
    def load_character_from_alphabet(
        x_path: str,
        y_path: str
    ) -> List[Tuple[List[float]]]:
        x_raw = np.load(x_path)
        y_raw = np.load(y_path)

        x_flat = x_raw.reshape(len(x_raw), -1)

        dataset = []
        for i in range(len(x_flat)):
            dataset.append((x_flat[i].tolist(), y_raw[i].tolist()))

        return dataset