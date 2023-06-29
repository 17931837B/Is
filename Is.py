#numpy:多次元配列や行列演算を効率的に処理するための計算ライブラリ
import numpy as np
#cv2:OpenCVをPythonのコードで使用するためのモジュール
import cv2
# matplotlib.pyplot:グラフや図を描画するためのライブラリ
import matplotlib.pyplot as plt
#sklearn:scikit-learn(サイキット-ラーン)と呼ばれる機械学習ライブラリ
import sklearn
#KMeans:クラスタリングアルゴリズム
from sklearn.cluster import KMeans
from PIL import Image
from sklearn.model_selection import train_test_split

# 設定パラメータ
img_n = 9

### npz用 ###
### データセットをnpzで作成した場合 ###
photos = np.load('./photo/224px_2.npz')  # 写真データを読み込み
x = photos['x']
y = photos['y']

# データセットを訓練用とテスト用に分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 分割後のデータのサイズを確認
print("Train data shape:", x_train.shape)
print("Test data shape:", x_test.shape)
print("Train label shape:", y_train.shape)
print("Test label shape:", y_test.shape)

# サンプルとしていくつかの画像をプロット
plt.figure(figsize=(12, 4))
for i in range(img_n):
    plt.subplot(3, 6, i+1)
    plt.imshow(x[i])
    plt.title(str(y[i]))
    plt.axis("off")
plt.tight_layout()
plt.show()

# 扱いやすいデータに整形・間引く
# ラベルが0か1のデータ(グーとパー)のみを扱う。訓練データは最大100個、テストデータは最大50個まで。
#訓練データ
train_x, train_y = [], []
for i in range(240):
    # データをまとめる(該当するデータをリストに追加していく)
    train_x.append(x_train[i])
    train_y.append(y_train[i])
    # 学習データは100個とする
    
#テストデータ
test_x, test_y = [], []
for i in range(60):
    test_x.append(x_test[i])
    test_y.append(y_test[i])
    # テストデータは50個
    
# numpyのarrayにまとめておく(多次元配列や行列演算を効率的に処理可能に)
train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

#xは画像データ、yはラベルデータ
print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)

# SIFTアルゴリズムで特徴量を計算するインスタンスを生成
sift_obj = cv2.SIFT_create()

#特徴量を追加するリスト
features_list = []

#SIFT特徴量を計算し、リストに追加
for x_image in train_x:
    # 画像をBGRからグレースケールに変換する(0～255の値一つ。3次元→1次元へ)
    gray = cv2.cvtColor(x_image, cv2.COLOR_BGR2GRAY)

    # グレースケール画像からSIFT特徴量を検出し、記述
    ## keypoints: キーポイントの位置・方向等の情報（分類では使わない）
    ## features：　キーポイント上の画像特徴量 (検出したKeyPointの数 x 128次元)、パワポスライド21ページ
    keypoints, features = sift_obj.detectAndCompute(gray, None)

    # SIFTの特徴量をリストに追加していく
    features_list.append(features)
    
# どんなKey Pointを取り出したか確認してみる
x_image = train_x[0]
kp_img = cv2.drawKeypoints(x_image, keypoints, None)
plt.axis("off")
plt.imshow(kp_img)

"""
# 画像毎に異なる数のKeypointが出力される
print('各画像のキーポイント数と次元数: ')
for idx, features in enumerate(features_list):
    print('画像', idx, '= ', features.shape)
"""
  
# 学習データの画像特徴量を一つのリストにまとめる
flat_features_list = []
for features in features_list:
    flat_features_list.extend(features)
#リストをNumPy配列に変換
flat_features_list = np.array(flat_features_list)
print('学習データの画像特徴量をまとめると: ', flat_features_list.shape)

# 特徴量をK個のクラスタにクラスタリング
# クラスタリングにはKMeansを使う
# KMeansのモデルのオブジェクトを作る(乱数のシード値を指定することで、実行ごとに結果が再現可能となる。)
n_clusters = 8
kmeans_obj = KMeans(n_clusters=n_clusters, random_state=0)

#KMeansモデルを用いてデータをクラスタリングし、クラスタ情報をkmeans_obj内に保管
kmeans_obj.fit(flat_features_list)

#  --> kmeans_objの中にクラスタ情報（＝単語情報)が入ったので、「単語づくり」は完了

#実行時に出る 「FutureWarning」は、KMeansの引数のデフォルト値(一部)が今後変更するお知らせ。今は気にしなくてOK。

hist_data_list = []
for features in features_list:
    # 画像毎の処理
    # features: (検出したKeyPointの数 x 128次元)

    # 各KeyPointのfeatureがどのクラスタに所属するかを求める
    #   -> clustsers：　(検出したKeyPointの数 x 1)
    #        値は所属するクラスタのIDが
    #        IDは、上の.fit()の段階で勝手に決められている
    clusters = kmeans_obj.predict(features)
    #下2行を実行すると、画像0(KeyPointの数 = 26)のクラスタ情報が表示される
    #print(clusters)
    #break
    # np.histogramで各クラスタの割当回数（=単語の出現回数)のデータに変換
    hist_data, _ = np.histogram(clusters, bins=n_clusters, range=(0, n_clusters))
    #print(hist_data)
    #break
    # データをリストに追加
    hist_data_list.append(hist_data)

# ヒストグラムデータを、BoVWでは特徴量として使う
img_feature_list = hist_data_list

# Numpy Arrayに変換しておく（この方が扱いやすい）
img_feature_list = np.array(img_feature_list)

print(img_feature_list[0])

#10個表示
for i in range(10):
  print(img_feature_list[i])
  
#　ここで、学習データを見直してみると

print('入力画像')
#　もともとのデータ
print('元の入力画像のshape: ', train_x.shape)
#　特徴量検出＋記述 と BoVW　を経て次元を削減(270000→8)
print('SIFT + BoVWで変換後: ', img_feature_list.shape)
print('')

print('ラベル')
# ラベルは特に変えてない
print('ラベルのshape', train_y.shape)
print('')

print('結果、{}次元のデータから1次元のデータを予測するっていう問題になった'.format(img_feature_list.shape[1]))

# 分類を行う
# SVMによる学習器を生成
svm_model = sklearn.svm.SVC()

# 学習データとそのクラスラベルを入力として受け取り、SVMモデルを学習させる。
svm_model.fit(img_feature_list, train_y)

# 学習したSVMモデルを使用して、学習データに対する予測を行う。(pred_trainはクラスラベル)
pred_train = svm_model.predict(img_feature_list)
# 実際のクラスラベル（train_y）と予測結果（pred_train）を比較し、予測の正確さ(正解率)を計算
accuracy_train = sklearn.metrics.accuracy_score(train_y, pred_train)
print(accuracy_train)

#手順は訓練データのときと同じ(train_xをtest_xに変えただけ)
#特徴量を追加するリスト
features_list = []

#SIFT特徴量を計算し、リストに追加
for x_image in test_x:
    # 画像をBGRからグレースケールに変換する
    gray = cv2.cvtColor(x_image, cv2.COLOR_BGR2GRAY)

    # グレースケール画像からSIFT特徴量を検出し、記述
    ## keypoints: キーポイントの位置・方向等の情報（分類では使わない）
    ## features：　キーポイント上の画像特徴量 (検出したKeyPointの数 x 128次元)
    keypoints, features = sift_obj.detectAndCompute(gray, None)

    # SIFTの特徴量をリストに追加していく
    features_list.append(features)
    
hist_data_list = []
for features in features_list:
    # 画像毎の処理
    # features: (検出したKeyPointの数 x 128次元)

    # 各KeyPointのfeatureがどのクラスタに所属するかを求める
    #   -> clustsers：　(検出したKeyPointの数 x 1)
    #        値は所属するクラスタのIDが
    #        IDは、上の.fit()の段階で勝手に決められている
    clusters =  kmeans_obj.predict(features)

    # np.histogramで各クラスタの割当回数（=単語の出現回数)のデータに変換
    hist_data, _ = np.histogram(clusters, bins=n_clusters, range=(0, n_clusters))

    # データをリストに追加
    hist_data_list.append(hist_data)

# ヒストグラムデータを、BoVWでは特徴量として使う
img_feature_list = hist_data_list

# Numpy Arrayに変換しておく（この方が扱いやすい）
img_feature_list = np.array(img_feature_list)

pred_test = svm_model.predict(img_feature_list)
accuracy_test = sklearn.metrics.accuracy_score(test_y, pred_test)
print(accuracy_test)