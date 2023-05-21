import cv2
import mediapipe

from utils.dataset_utils import load_dataset, load_reference_signs
from utils.mediapipe_utils import mediapipe_detection
from sign_recorder import SignRecorder
from webcam_manager import WebcamManager


if __name__ == "__main__":
    # Create dataset of the videos where landmarks have not been extracted yet
    videos = load_dataset()

    # Create a DataFrame of reference signs (name: str, model: SignModel, distance: int)
    reference_signs = load_reference_signs(videos)

    # Object that stores mediapipe results and computes sign similarities
    sign_recorder = SignRecorder(reference_signs)

    # Object that draws keypoints & displays results
    webcam_manager = WebcamManager()

    # Turn on the webcam
    cap = cv2.VideoCapture(0)
    # Set up the Mediapipe environment
    with mediapipe.solutions.holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # 여기서 캠에서 읽어온 이미지와 라벨정보를 props로 넘긴다
            image, results = mediapipe_detection(frame, holistic)

            
            # 결과 값 예측 시작
            sign_detected = sign_recorder.process_results(results)

            # 예측된 결과 카메라에 반영
            webcam_manager.update(frame, results, sign_detected)

            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord("q"):  # q누르면 종료
                break

        cap.release()
        cv2.destroyAllWindows()


        # 0. 결과 값을 return할 수 있는 String으로 추출해보기 (O)

        # 1. 일단 더미데이터를 넣어서 결과값이 추출가능한지 수정해보기 (?)
        
        # 2. record모드와 상관없이 항상 데이터를 뽑아서 보내지는지 확인해보기 (O)
        
        # 3. 프론트에서 보내고 있는 배열값을 해당 함수에서 사용하고있는 방식으로 파싱할 것

        # 4. 동시에 카메라가 아니라 한번의 api호출에 대한 결과 값 return으로 함수처리 수정하기 

        # 5. 웹소켓 연결 및 실제 api연동해보기

        # 6. 영상을 기준으로 모델을 만드는 것 같은데 파일 기준으로 만들 수 있는지 찾아보기