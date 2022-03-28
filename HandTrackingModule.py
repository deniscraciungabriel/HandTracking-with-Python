from pydoc import ModuleScanner
import cv2
import mediapipe as mp


class handDetector:
    def __init__(self, mode=False, max=2, detectionConf=0.2, trackCon=0.2):
        self.mode = mode
        self.max = max
        self.detectionConf = detectionConf
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            # self.mode, self.max, self.detectionConf, self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def drawHands(
        self,
        img,
    ):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for i in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, i, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, finger=0):

        lmPos = []

        if self.results.multi_hand_landmarks:
            for i in self.results.multi_hand_landmarks:
                for id, lm in enumerate(i.landmark):
                    (
                        h,
                        w,
                        c,
                    ) = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmPos.append([id, cx, cy])

        if finger != 0 and len(lmPos) > 0:
            return lmPos[finger]
        else:
            return lmPos


"""cap = cv2.VideoCapture(0)
tips = [4, 8, 12, 16, 20]
detector = handDetector()

while True:
    success, img = cap.read()
    img = detector.drawHands(img)

    lmPos = detector.findPosition(img)
    print(lmPos)

    cv2.imshow("IMAGE", img)
    cv2.waitKey(1)"""

cap = cv2.VideoCapture(0)
tips = [4, 8, 12, 16, 20]
detector = handDetector()

while True:
    success, img = cap.read()
    img = detector.drawHands(img)

    index = detector.findPosition(img, 8)
    thumb = detector.findPosition(img, 4)
    pink = detector.findPosition(img, 20)
    if thumb and pink:
        cv2.line(
            img,
            (thumb[1], thumb[2]),
            (pink[1], pink[2]),
            (255, 0, 0),
            3,
        )
        cv2.circle(
            img,
            (int((thumb[1] + pink[1]) / 2), int((thumb[2] + pink[2]) / 2)),
            10,
            (0, 0, 255),
            cv2.FILLED,
        )
        cv2.circle(
            img,
            (thumb[1], thumb[2]),
            5,
            (0, 255, 0),
            cv2.FILLED,
        )
        cv2.circle(
            img,
            (pink[1], pink[2]),
            5,
            (0, 255, 0),
            cv2.FILLED,
        )
        cv2.circle(
            img,
            (index[1], index[2]),
            5,
            (0, 255, 0),
            cv2.FILLED,
        )

    cv2.imshow("IMAGE", img)
    cv2.waitKey(1)
