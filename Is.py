import cv2
import matplotlib.pyplot as plt

def capture_and_plot():
    cap = cv2.VideoCapture(0)  # カメラをキャプチャする

    while True:
        ret, frame = cap.read()  # フレームを読み込む

        cv2.imshow('Camera', frame)  # フレームを表示する

        # Enterキーが押されたら画像を保存してプロットする
        if cv2.waitKey(1) == 13:  # 13はEnterキーのキーコード
            cv2.imwrite('captured_image.jpg', frame)  # 画像を保存する
            break

    cap.release()  # カメラを解放する
    cv2.destroyAllWindows()  # ウィンドウを閉じる

    # 保存した画像をプロットする
    img = cv2.imread('captured_image.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


capture_and_plot()
