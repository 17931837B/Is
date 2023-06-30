import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_max_value_position(image_path):
    # 画像を読み込む
    img = cv2.imread(image_path)

    # 画像の高さと幅を取得
    height, width, _ = img.shape
    print(height)
    print(width)

    # 各ピクセルごとに計算した値を得る
    r = img[:,:,2].astype(int)
    g = img[:,:,1].astype(int)
    b = img[:,:,0].astype(int)
    values = r*2 - (g + b)

    # 最大値のインデックスを求める
    max_index = np.unravel_index(np.argmax(values), values.shape)

    print("最大値の位置（行, 列）:", height-max_index[0], width-max_index[1])
    max_position = (max_index[1], max_index[0])

    # 画像をプロットして最大値の位置に目印を追加
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.scatter(max_position[0], max_position[1], color='red', marker='x')
    plt.axis('off')
    plt.show()

def capture_and_calculate():
    cap = cv2.VideoCapture(0)  # カメラをキャプチャする

    while True:
        ret, frame = cap.read()  # フレームを読み込む

        cv2.imshow('Camera', frame)  # フレームを表示する

        # Enterキーが押されたら画像を保存して計算する
        if cv2.waitKey(1) == 13:  # 13はEnterキーのキーコード
            cv2.imwrite('captured_image.jpg', frame)  # 画像を保存する
            break

    cap.release()  # カメラを解放する
    cv2.destroyAllWindows()  # ウィンドウを閉じる

    # 保存した画像の最大値の位置を求めてプロットする
    find_max_value_position('captured_image.jpg')

capture_and_calculate()
