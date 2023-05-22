import pandas as pd
import numpy as np
from collections import Counter
from utils.dtw import dtw_distances
from models.sign_model import SignModel



class SignRecorder(object):
    def __init__(self, reference_signs: pd.DataFrame):

        # List of results stored each frame
        self.recorded_results = []

        # DataFrame storing the distances between the recorded sign & all the reference signs from the dataset
        self.reference_signs = reference_signs


    def process_results(self, results) -> (str, bool):
        

        self.recorded_results = results
        self.compute_distances()
        return self._get_sign_predicted()

    def compute_distances(self):

        left_hand_list, right_hand_list = [], []
        left_hand_list = self.recorded_results["left"]
        right_hand_list = self.recorded_results["right"]

        # print("왼쪽 좌표 추출")
        # print(left_hand_list)
        # print("오른쪽 좌표 추출")
        # print(right_hand_list)


        recorded_sign = SignModel(left_hand_list, right_hand_list)

        self.reference_signs = dtw_distances(recorded_sign, self.reference_signs)


    def _get_sign_predicted(self, batch_size=5, threshold=0.5):

        # Get the list (of size batch_size) of the most similar reference signs
        sign_names = self.reference_signs.iloc[:batch_size]["name"].values
        # print("sign_name")
        # print(sign_names)

        # Count the occurrences of each sign and sort them by descending order
        sign_counter = Counter(sign_names).most_common()
        # print("sign_counter")
        # print(sign_counter)
     
        self.reference_signs["distance"].values[:] = 0


        predicted_sign, count = sign_counter[0]
        # if count / batch_size < threshold:
        #     return "Signe inconnu"
        return predicted_sign
