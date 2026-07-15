# Lesson 5.5: 나만의 딥러닝 프로젝트 시작하기 (데이터셋과 실무 리소스 가이드)

---

대망의 마지막 강의에 오신 것을 환영합니다! 지금까지 우리는 딥러닝이 무엇인지, 어떻게 작동하는지, 그리고 TensorFlow(Keras)와 PyTorch라는 두 가지 강력한 무기를 사용해 최첨단 신경망 모델을 설계하는 방법까지 모두 마스터했습니다. 

이제 여러분은 남이 만들어 놓은 예제 코드를 따라 치는 수준을 넘어, **'세상에 존재하는 진짜 문제'**를 딥러닝으로 해결할 준비가 되었습니다. 이 마지막 파트에서는 나만의 프로젝트를 시작하기 위해 어디서 데이터를 구해야 하는지, 그리고 실무 모델링 기법과 공부 방향을 아주 구체적이고 상세하게 안내해 드리겠습니다.

---

## 1. 첫 번째 독립 프로젝트 추천: Fashion MNIST 정복하기

수기 숫자(MNIST) 데이터셋은 딥러닝계의 'Hello World'입니다. 하지만 너무 쉽다는 단점이 있죠. 우리가 만든 단순한 모델로도 99%의 정확도가 훌쩍 넘어가 버리니까요. 이제 여러분이 직접 부딪혀볼 첫 번째 챌린지로 **`Fashion MNIST`**를 추천합니다.

### 👕 Fashion MNIST란?
*   **특징**: 손글씨 숫자 0~9 대신, 티셔츠, 바지, 드레스, 스니커즈, 발목 부츠(Ankle Boot) 등 **10가지 종류의 옷과 신발 사진**으로 이루어진 데이터셋입니다.
*   **크기**: 기존 MNIST와 완전히 동일합니다. (28x28 픽셀 흑백 이미지, 훈련용 60,000장, 검증용 10,000장)
*   **난이도**: 숫자를 구별하는 것보다 옷의 미세한 패턴(예: 풀오버 vs 코트)을 구별하는 것이 훨씬 어렵습니다. 여기서 **정확도 92%를 넘기면 딥러닝의 기본기를 완벽히 갖춘 것**이며, **94%를 넘긴다면 대단히 훌륭한 수준(Impressive)**이라고 자부하셔도 좋습니다!

### 💻 실무 적용 코드 (Fashion MNIST 분류기 만들기)
데이터만 갈아 끼우면 기존에 짜두었던 `LeNet` 아키텍처를 그대로 활용할 수 있습니다. 

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np

# 1. 패션 MNIST 데이터 불러오기 (이 한 줄만 바꾸면 됩니다!)
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_valid, y_valid) = fashion_mnist.load_data()

# 데이터 전처리 (0~255 픽셀 값을 0~1 사이로 정규화 및 차원 추가)
X_train = X_train.reshape(60000, 28, 28, 1).astype('float32') / 255
X_valid = X_valid.reshape(10000, 28, 28, 1).astype('float32') / 255

# 2. LeNet 스타일의 CNN 모델 아키텍처 구성
model = Sequential([
    # 이미지 특징을 뽑아내는 합성곱 층
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # 1차원으로 펴서 밀집층으로 전달
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax') # 10개의 옷 종류 분류
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 3. 모델 훈련 (시간 관계상 1 에폭만 돌리지만 실무에선 Early Stopping을 쓰세요!)
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=1, batch_size=128)

# 4. 실전 추론 (Inference) 테스트
# 검증 데이터셋의 첫 번째 이미지(발목 부츠, 정답 레이블=9)를 모델이 맞춰보게 합니다.
predictions = model.predict(X_valid[0:1])
predicted_class = np.argmax(predictions[0])
confidence = np.max(predictions[0]) * 100

print(f"모델의 예측: {predicted_class}번 클래스 (자신감: {confidence:.1f}%)")
# 모델이 9번 클래스(발목 부츠)라고 98.4%의 확신을 가지고 대답한다면 성공입니다!
```
여러분은 이 코드를 바탕으로 레이어를 늘려보거나, 드롭아웃을 추가하고, 하이퍼파라미터를 튜닝하며 '마의 94% 벽'을 깨는 도전을 해보시길 강력히 권장합니다.

---

## 2. 야생의 실전 데이터셋 구하기

Fashion MNIST를 정복했다면, 이제 진짜 '돈이 되는' 실전 데이터셋을 구해서 프로젝트를 할 차례입니다. 데이터 사이언티스트들이 애용하는 3대 보물창고를 소개합니다.

1.  **Kaggle (캐글, kaggle.com)**:
    세계 최대의 데이터 과학 경진대회 플랫폼입니다. 프랑스의 거대 이커머스 기업인 CDiscount가 상품 이미지 분류 모델을 만드는 사람에게 약 4천만 원($35,000)의 상금을 걸었던 것처럼, 수많은 기업들이 상금과 함께 고품질의 실무 데이터를 올려둡니다. 대회 기한이 끝났더라도 데이터는 계속 다운로드할 수 있으므로 최고의 학습장입니다.
2.  **Figure Eight (구 CrowdFlower)**:
    수많은 사람들이 직접 수작업으로 라벨링(정답 달기)을 해둔 초고품질의 데이터셋을 제공합니다. (사이트의 'Data for Everyone' 섹션에서 image를 검색해 보세요.)
3.  **Luke de Oliveira의 큐레이션 (bit.ly/LukeData)**:
    유명 딥러닝 연구자가 실무자들이 꼭 알아야 할 최고의 컴퓨터 비전 및 자연어 처리 데이터셋들을 깔끔하게 리스트업해 둔 곳입니다.

---

## 3. 내가 가진 데이터로 딥러닝 하기 (Wide & Deep 아키텍처)

아마 회사에서 이미 기존에 머신러닝(SVM이나 선형 회귀 등)을 돌리던 '표(Table)' 형태의 데이터가 있을 수도 있습니다. 이런 데이터도 딥러닝에 넣을 수 있을까요? 당연합니다!

구글(Google)의 연구원들은 한발 더 나아가, **기존에 인간이 예쁘게 가공해 둔 특성(Engineered Features)**과 **가공하지 않은 날것의 데이터(Raw Inputs)**를 동시에 딥러닝 모델에 집어넣는 **`Wide & Deep Modeling`**이라는 기법을 대중화시켰습니다.

아래 다이어그램을 보시죠.

![Wide & Deep Learning 아키텍처 도식화](/Users/shinwookkang/.gemini/antigravity/brain/cb5306ae-e4e0-4381-a3ab-20a49944023c/wide_and_deep_architecture_1784099326906.jpg)

### 🧠 Wide & Deep 모델의 철학
*   **Deep Part (깊은 부분)**: 왼쪽의 이미지나 텍스트 같은 날것의 데이터(Raw Inputs)를 깊은 신경망(Hidden Layers)에 밀어 넣습니다. 모델은 여기서 스스로 복잡한 특징(Feature)을 족집게처럼 찾아냅니다. (이것을 '일반화, Generalization'라고 합니다.)
*   **Wide Part (넓은 부분)**: 전문가인 우리가 보기에 '이건 무조건 정답에 영향을 미쳐!'라고 생각해서 미리 계산해 둔 특성(예: 유저의 성별 x 상품 카테고리 조합)을 딥러닝을 거치지 않고 바로 꽂아 넣습니다. (이것을 '암기, Memorization'라고 합니다.)
*   **Concatenate (결합)**: 두 개의 출력을 나란히 이어 붙여서(Catenate) 최종 예측을 내립니다.

### 💻 실무 Keras 코드 (함수형 API 사용)
이런 두 갈래 길을 만들려면 기존의 `Sequential`(순차적) 방법 대신, **Keras의 함수형 API(Functional API)**를 사용해야 합니다.

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model

# 1. Wide 부분: 사람이 만든 데이터 (예: 5개의 핵심 수치 데이터)
input_wide = Input(shape=[5], name="wide_engineered_features")

# 2. Deep 부분: 날것의 데이터 (예: 100개의 픽셀 또는 단어 데이터)
input_deep = Input(shape=[100], name="deep_raw_inputs")

# Deep 데이터는 스스로 특징을 찾도록 은닉층을 2개 거치게 합니다.
hidden1 = Dense(32, activation="relu")(input_deep)
hidden2 = Dense(32, activation="relu")(hidden1)

# 3. 마법의 결합! (Concatenate)
# 사람이 만든 Wide 데이터와, AI가 깊게 파고든 Deep 데이터를 하나로 묶습니다.
concat = Concatenate()([input_wide, hidden2])

# 4. 최종 출력층
output = Dense(1, activation="sigmoid")(concat)

# 5. 모델 조립 (입력 문이 2개인 모델 완성!)
model = Model(inputs=[input_wide, input_deep], outputs=output)

model.summary() # 이 코드를 실행해보면 두 갈래 길이 합쳐지는 구조를 볼 수 있습니다.
```
이 구조는 현재 유튜브 알고리즘이나 구글 플레이스토어 앱 추천 시스템 등 현업의 **추천 시스템(Recommendation System)**에서 가장 널리 쓰이는 형태입니다!

---

## 4. 인류를 구원할 딥러닝 프로젝트 (사회적 가치 창출)

강의를 마무리하며, 여러분이 배운 이 기술을 어디에 쓸 것인지 방향성을 제시해 드리고자 합니다. 강사의 개인 사이트(`johncrone.com/resources`)에 가시면 수많은 튜토리얼과 논문 구현체(GitXiv)를 볼 수 있지만, 그중에서도 **'풀 가치가 있는 문제들(Problems Worth Solving)'** 섹션을 꼭 읽어보시길 권합니다.

1.  **McKinsey Global Institute의 10대 사회 문제**:
    평등, 교육, 보건, 안보, 위기 대응, 환경 등 10가지 인류의 난제를 딥러닝이 어떻게 해결할 수 있는지 분석한 보고서입니다. 놀랍게도 우리가 배운 Dense Net과 CNN(합성곱 신경망) 기술만으로도 이 10가지 문제의 대부분을 해결하는 데 직접적으로 기여할 수 있습니다.
2.  **기후 변화와 머신러닝 (Climate Change & Machine Learning)**:
    알파고를 만든 딥마인드(DeepMind)의 데미스 하사비스(Demis Hassabis)와 요슈아 벤지오(Yoshua Bengio) 교수가 공동 집필한 55페이지짜리 명저입니다. 전력 시스템 효율화, 친환경 건축, 산림 보존, 기후 예측 등 컴퓨터 비전 모델이 지구를 구하는 데 어떻게 쓰일 수 있는지 상세히 나열되어 있습니다.

여러분은 이제 세상을 바꿀 수 있는 지팡이를 손에 쥐었습니다. 캐글에서 상금을 타는 것도 좋지만, 기회가 된다면 인류와 사회를 위해 기여할 수 있는 프로젝트에 이 기술을 적용해 보시길 진심으로 응원합니다. 

기나긴 딥러닝 여정을 끝까지 함께 해주셔서 감사합니다. 여러분의 무한한 건승을 기원합니다!
