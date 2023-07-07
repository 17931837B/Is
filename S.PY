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
    
    x = width-max_index[1]
    y = height-max_index[0]
    

    print("撮影におけるx座標", x)
    print("撮影におけるy座標", y)

    # 画像をプロットして最大値の位置に目印を追加
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.scatter(max_position[0], max_position[1], color='red', marker='x')
    plt.axis('off')
    plt.show()

    return max_position

def capture_and_calculate():

    cap = cv2.VideoCapture(0)  # カメラをキャプチャする
    while True:
        ret, frame = cap.read()  # フレームを読み込む

        cv2.imshow('Camera', frame)  # フレームを表示する

        # Enterキーが押されたら画像を保存して計算する
        if cv2.waitKey(1) == 13:  # 13はEnterキーのキーコード
            cv2.imwrite('captured_image1.jpg', frame)  # 画像を保存する
            break
    cap.release()  # カメラを解放する


    cap = cv2.VideoCapture(0)  # カメラをキャプチャする
    while True:
        ret, frame = cap.read()  # フレームを読み込む

        cv2.imshow('Camera', frame)  # フレームを表示する

        # Enterキーが押されたら画像を保存して計算する
        if cv2.waitKey(1) == 13:  # 13はEnterキーのキーコード
            cv2.imwrite('captured_image2.jpg', frame)  # 画像を保存する
            break
    cap.release()  # カメラを解放する


    cv2.destroyAllWindows()  # ウィンドウを閉じる

    # 保存した画像の最大値の位置を求めてプロットする
    max_pos1 = find_max_value_position('captured_image1.jpg')
    max_pos2 = find_max_value_position('captured_image2.jpg')

    diff_x = abs(max_pos2[0] - max_pos1[0])
    diff_y = abs(max_pos2[1] - max_pos1[1])
    
    print("x方向の差:", diff_x)
    print("y方向の差:", diff_y)

    return diff_x, diff_y


from scipy.signal import windows
from scipy.io import wavfile
import numpy as np
from playsound import playsound

IN_WAVE_FILE = "./IS音声ファイル/富嶽三十六景.wav"   # 16bit モノラル音声（前提）
OUT_WAVE_FILE = "out.wav"

PITCH = 1.0                  # ピッチの倍率
N_TERM = 30                    # 標本化定理の近似項数
RATE = 1.0           # 再生速度の倍率

def resampling(data_in):
    """ リサプリング （音を高く/低くするが再生時間も変わる）"""

    # wavの読み込み
    data_in = data_in.astype(np.float64)
    n_samples = len(data_in)

    n_samples_out = int(n_samples / PITCH)
    data_out = np.zeros(n_samples_out)

    for n in range(n_samples_out):
        analog_time = PITCH * n          # アナログ信号の時刻 t を求める
        digital_time = int(analog_time)  # 整数値に変換

        # 標本化定理に基づくアナログ信号復元によるリサンプリング
        sum_index = np.arange(digital_time - N_TERM // 2,
                              digital_time + N_TERM // 2 + 1)
        start_index = np.min(np.where(sum_index >= 0))
        end_index = np.max(np.where(sum_index <= n_samples))
        sinc_vector = np.sinc(analog_time - sum_index[start_index:end_index])
        data_out[n] = data_in[sum_index[start_index:end_index]].dot(sinc_vector)

    return data_out


def calc_autocorr(wave_data, corr_size, lag):
    """ 自己相関関数を計算する """

    autocorr = 0.0
    for i in range(corr_size):
        autocorr += wave_data[i] * wave_data[i + lag]

    return autocorr


def get_period(wave_data, period_min, period_max, corr_size):
    """ 相関関数のピーク位置に基づいて周期を計算する """

    corr_max = 0.0
    period = period_min

    for p in range(period_min, period_max):

        corr = calc_autocorr(wave_data, corr_size, p)  # 相関関数の計算

        # 相関関数のピークを求め、そのピーク位置を周期として取得する
        if corr > corr_max:
            corr_max = corr     # 相関関数のピーク値を更新
            period = p          # 周期を取得

    return period


def time_stretch(data_in, fs):
    """ タイムストレッチ (音の高さを変えずに再生時間を変更する) """

    # wavの読み込み
    data_in = data_in.astype(np.float64)
    n_samples = len(data_in)

    corr_size = int(fs * 0.01)    # 相関関数のサイズ 10ms
    min_period = int(fs * 0.005)  # 窓内における音データの周期の最小値 5ms
    max_period = int(fs * 0.02)   # 窓内における音データの周期の最大値 20ms

    offset_in = 0                # オーバーラップアドの開始位置 (入力側)
    offset_out = 0               # オーバーラップアドの開始位置 (出力側)

    data_out = np.zeros(int(n_samples / RATE) + 1)
    while (offset_in + max_period * 2 < n_samples):
        # 窓内の音データに対して(窓サイズ corr_size)、
        # 相関関数を利用することで音の周期を計算する
        period = get_period(data_in[offset_in:],
                            min_period, max_period, corr_size)

        if RATE >= 1.0:  # fast
            # オーバーラップアド（時間を縮める）
            window = windows.hann(2 * period)
            for n in range(period):
                data_out[offset_out + n] = \
                    data_in[offset_in + n] * window[period + n]
                data_out[offset_out + n] += \
                    data_in[offset_in + period + n] * window[n]
            del window

            # オーバーラップアドしていない音データをそのままコピー
            q = int(period / (RATE - 1.0) + 0.5)
            for n in range(period, n_samples):
                if n >= q:
                    break
                elif offset_in + period + n >= n_samples:
                    break
                else:
                    data_out[offset_out + n] = data_in[offset_in + period + n]

            offset_in += period + q
            offset_out += q
        else:  # slow
            # オーバーラップアドしていない音データをそのままコピー
            data_out[offset_out: offset_out + period] = \
                data_in[offset_in: offset_in + period]

            # オーバーラップアド（時間を伸ばす）
            window = windows.hann(2 * period)
            for n in range(period):
                data_out[offset_out + period + n] = \
                    data_in[offset_in + n] * window[n]
                data_out[offset_out + period + n] += \
                    data_in[offset_in + period + n] * window[period + n]
            del window

            # オーバーラップアドしていない音データをそのままコピー
            q = int(period * RATE / (1.0 - RATE) + 0.5)
            for n in range(period, n_samples):
                if n >= q:
                    break
                elif offset_in + period + n >= n_samples:
                    break
                else:
                    data_out[offset_out + period + n] = data_in[offset_in + n]

            offset_in += q
            offset_out += period + q

    return data_out


def pitch_shift():
    """ 再生時間を変えずに音の高さを変える """

    # wavの読み込み
    fs, data_in = wavfile.read(IN_WAVE_FILE)

    # リサンプリング
    data_out = resampling(data_in)

    # タイムストレッチ
    data_out = time_stretch(data_out, fs)

    # wavの書き込み
    data_out = data_out.astype(np.int16)
    wavfile.write(OUT_WAVE_FILE, fs, data_out)

jyouken = 0



if __name__ in '__main__':

    diff_x, diff_y = capture_and_calculate()

    if 0 <= diff_x <= 100:
        jyouken = 1

        if jyouken == 1:
            playsound("./IS音声ファイル/富嶽三十六景.wav")
        else:
            RATE = 2.0
            PITCH = 0.5
            pitch_shift()
            playsound("out.wav")





