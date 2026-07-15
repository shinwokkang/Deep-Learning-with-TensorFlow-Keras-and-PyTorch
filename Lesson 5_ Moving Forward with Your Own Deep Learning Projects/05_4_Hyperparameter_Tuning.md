# Lesson 5.4: 딥러닝 모델의 영혼 불어넣기 (Hyperparameter Tuning 7단계 마스터 가이드)

---

## 1. 하이퍼파라미터(Hyperparameter)란 무엇인가요?

우리가 지금까지 설계한 딥러닝 모델은 수만 개의 가중치(Weights)와 편향(Biases)을 가지고 있습니다. 이 값들은 모델이 데이터를 보면서 **스스로 학습해서 찾아내는 숫자**들입니다. 이를 **'파라미터(Parameter)'**라고 부릅니다.

하지만 모델이 스스로 정할 수 없는 숫자들이 있습니다.
*   "은닉층을 몇 개로 할까?"
*   "뉴런을 64개로 할까, 128개로 할까?"
*   "학습률(Learning Rate)은 0.001로 할까, 0.01로 할까?"
*   "미니배치는 몇 장씩 묶을까?"

이처럼 **'인간(개발자)이 모델을 훈련시키기 전에 미리 세팅해 주어야 하는 설정값'**들을 **'하이퍼파라미터(Hyperparameter)'**라고 부릅니다. 이 하이퍼파라미터를 어떻게 조율(Tuning)하느냐에 따라 엉망진창이던 모델이 세계 최고의 모델로 둔갑하기도 합니다.

하지만 무턱대고 아무 숫자나 바꿔보는 것은 모래사장에서 바늘 찾기입니다. 이번 강의에서는 실무에서 모델의 성능을 극대화하기 위해 밟아야 하는 **'하이퍼파라미터 튜닝 7단계 가이드라인'**을 아주 상세하게 파헤쳐 보겠습니다.

*(주의: 이 7단계는 절대적인 법이 아닙니다. 7번까지 갔다가 다시 3번으로 돌아오는 등, 실험을 거듭하며 수십 번 뺑뺑이를 도는 것이 실무의 일상입니다!)*

---

## 2. 하이퍼파라미터 튜닝 7단계 가이드

### 🛠️ 단계 1: 파라미터 초기화 (Parameter Initialization)
*   **개념**: 아무것도 모르는 아기 모델의 뇌 세포(가중치)에 맨 처음 어떤 숫자를 넣어주고 시작할 것인가의 문제입니다. 맨 처음 숫자를 잘못 넣으면 모델이 평생 학습을 못 하고 바보가 될 수 있습니다.
*   **실무 팁**: 과거에는 이 초기값을 찾는 게 큰 연구 주제였지만, 현재는 걱정할 필요가 없습니다. **Keras나 PyTorch가 내부적으로 'Xavier(Glorot) 초기화'나 'He 초기화'라는 수학적으로 가장 완벽한 방식을 알아서 세팅**해 주기 때문입니다. 편향(Bias)은 0으로 시작하는 것이 기본입니다. (우리는 그냥 라이브러리를 믿고 넘어가면 됩니다!)

### 🛠️ 단계 2: 손실 함수(Cost Function) 선택
*   **개념**: 모델이 얼마나 틀렸는지 점수를 매기는 채점관을 고르는 단계입니다.
*   **실무 팁**: 목적에 따라 딱 정해져 있습니다.
    *   **분류 문제**(고양이 vs 강아지, 숫자 0~9 맞추기): 무조건 **`Cross-Entropy Loss`**를 씁니다.
    *   **회귀 문제**(내일의 주가 맞추기, 집값 예측하기 등 연속된 숫자 맞추기): 무조건 **`MSE (Mean Squared Error, 평균 제곱 오차)`**를 씁니다.

### 🛠️ 단계 3: 찍는 것(Chance)보다 나은 상태 만들기
*   **개념**: 모델을 처음 돌려봤는데 정확도가 10% 미만으로 나온다면? (숫자가 0부터 9까지 10개인데 정답률이 10%라는 것은 그냥 눈감고 연필을 굴려서 찍는 것과 똑같다는 뜻입니다.) 모델이 심각한 병에 걸린 상태입니다.
*   **해결책 (문제 단순화하기)**:
    1.  **데이터 줄이기**: 0~9까지 10개를 분류하지 말고, 일단 0과 1 두 개만 놓고 분류해 보세요. 그래도 못 맞춘다면 코드 자체에 심각한 버그가 있는 것입니다.
    2.  **구조 단순화**: 모델이 너무 깊어서 오히려 학습 신호가 전달되지 않는 '기울기 소실(Vanishing Gradient)'일 수 있습니다. 레이어를 확 줄이고 얕은 신경망(Shallow Net)부터 다시 시작하세요.
    3.  **샘플 줄이기**: 전체 데이터를 다 돌리면 한 번 테스트하는 데 너무 오래 걸립니다. 6만 장 대신 1천 장만 가지고 모델이 제대로 학습되는지 빠르게 실험(Iteration)해 보세요.

### 🛠️ 단계 4: 레이어(Layer) 디자인 맞추기
*   **개념**: 모델이 찍는 수준을 벗어나 학습을 시작했다면, 본격적으로 뇌의 구조를 디자인할 차례입니다.
*   **실무 팁**:
    *   **층(Depth)의 수**: 층을 하나씩 늘리거나 지워가며 성능을 봅니다.
    *   **뉴런(Width)의 수**: 32, 64, 128, 256처럼 **'2의 제곱수'**로 뉴런 수를 늘려보세요. (컴퓨터 메모리 구조상 2의 제곱수가 계산 속도가 가장 빠릅니다.)
    *   **레이어 종류 변경**: 이미지 문제인데 단순히 `Dense(Linear)`만 쓰고 있나요? 당장 `Conv2D(합성곱)` 레이어로 바꿔보세요. 성능이 차원이 다르게 폭발할 것입니다.

### 🛠️ 단계 5: 과적합(Overfitting) 방지하기
*   **개념**: 훈련 점수는 높은데, 처음 보는 시험지(Test Data)만 보면 쩔쩔매는 과적합 상태를 치료해야 합니다.
*   **해결책**:
    1.  **Dropout / BatchNorm 추가**: 뉴런의 일부를 기절시키거나 배치를 정규화하여 훈련을 혹독하게 만듭니다.
    2.  **데이터 증강(Data Augmentation)**: 고양이 사진을 뒤집고, 늘리고, 회전시켜서 억지로 데이터를 뻥튀기합니다. 데이터가 많아질수록 과적합은 줄어듭니다.
    3.  **조기 종료 (Early Stopping) / 가중치 복원**: 검증 오차(Validation Loss)가 떨어지다가 다시 치솟는 그 지점! 그 타이밍의 가중치로 타임머신을 타고 되돌아가야 합니다. (아래 4번 목차에서 실제 코드로 배울 것입니다.)

### 🛠️ 단계 6: 학습률 (Learning Rate) 조절
*   **개념**: 최적화 요정이 산을 내려갈 때 보폭의 크기입니다. 보폭이 너무 크면 산꼭대기로 튕겨 나가고, 너무 작으면 평생 산을 못 내려옵니다.
*   **실무 팁**: 과거에는 이 숫자를 0.01, 0.005 조절하느라 밤을 새웠습니다. 하지만 지금은 **`Adam`, `Nadam`, `RMSprop`** 같은 똑똑한 최적화 요정(Optimizer)들이 산의 가파름을 보고 알아서 보폭을 자동 조절해 줍니다. 따라서 실무에서는 그냥 Adam을 믿고 학습률 튜닝은 크게 신경 쓰지 않는 추세입니다.

### 🛠️ 단계 7: 배치 크기 (Batch Size) 설정
*   **개념**: 한 번에 몇 장의 사진을 보고 가중치를 고칠 것인가입니다.
*   **실무 팁**: 가장 성능에 영향이 적은 하이퍼파라미터입니다. 제일 나중에 손대세요.
    *   기본적으로 **32**로 시작하세요.
    *   속도를 높이고 싶다면 64, 128까지는 늘려도 좋습니다.
    *   **128 이상으로 올리는 것은 비추천합니다.** 배치가 너무 커지면 모델이 학습의 딜레마(지역 최소점, Local Minima)에 빠져버릴 확률이 높아집니다. 
    *   만약 컴퓨터 그래픽카드 메모리(VRAM)가 터진다면(OOM 에러), 어쩔 수 없이 16, 8, 4로 절반씩 줄여나가야 합니다.

---

## 3. 하이퍼파라미터 튜닝을 '자동화'하는 방법

개발자들은 게으릅니다. 저 수많은 하이퍼파라미터 조합(레이어 64개 썼다가 128개 썼다가, 드롭아웃 0.3 줬다가 0.5 줬다가...)을 언제 다 손으로 치고 앉아있을까요? 그래서 이것을 대신해 주는 자동화 도구들이 탄생했습니다. (예: `Hyperopt`, Keras 전용인 `Talos` 등)

이 도구들에게 "뉴런 수는 32~256 사이에서 알아서 찾아보고, 드롭아웃은 0.1~0.5 사이에서 찾아봐!"라고 범위를 주면 컴퓨터가 알아서 최적의 조합을 찾아냅니다.

이때 컴퓨터가 조합을 찾는 방식에는 크게 두 가지가 있습니다. 아래 이미지를 먼저 보시죠.

![그리드 서치 vs 랜덤 서치 비교](/Users/shinwookkang/.gemini/antigravity/brain/cb5306ae-e4e0-4381-a3ab-20a49944023c/grid_vs_random_search_1784097620940.jpg)

### ❌ 격자 탐색 (Grid Search) - "인간의 멍청한 강박관념"
왼쪽 그림입니다. 인간은 깔끔한 걸 좋아합니다. 그래서 `A=[10, 20, 30]`, `B=[0.1, 0.2, 0.3]` 이렇게 바둑판의 교차점처럼 딱딱 떨어지는 격자(Grid) 형태로 모든 조합을 테스트하려고 합니다.
문제는, 진짜 최고의 정답(노란 별표)이 15나 25 같은 애매한 위치에 뾰족하게 솟아 있다면, 바둑판 교차점만 뒤지는 격자 탐색은 이 최고점을 평생 놓치게 됩니다.

### ✅ 무작위 탐색 (Random Search) - "통계학의 승리"
오른쪽 그림입니다. 제임스 베르그스트라와 요슈아 벤지오 교수가 논문으로 증명한 방법입니다. 그냥 주어진 범위 내에서 숫자를 **완전 무작위(Random)로 흩뿌려서 테스트하는 것**이 오히려 바둑판처럼 촘촘히 하는 것보다 **압도적으로 빠르게, 더 뾰족한 정답(최적점)을 찾아낼 확률이 높다**는 사실이 밝혀졌습니다.

따라서 자동화 툴을 쓰실 때는 무조건 `Random Search`나 통계적 확률을 이용하는 `Bayesian Optimization(베이지안 최적화)` 방식을 선택하셔야 합니다!

---

## 4. [실전 딥다이브] 실무 코드로 배우는 '과적합 방지(Step 5)' 템플릿

다른 하이퍼파라미터는 자동화 툴을 쓰면 되지만, **'과적합을 방지하고 가장 똑똑했던 시절의 기억으로 되돌리는 기술(Early Stopping)'**은 실무 코드에 무조건 수동으로 세팅을 해두어야 합니다. 우리가 이전 단원들에서 만들었던 `Deep Net` 코드 위에 이 기술을 어떻게 얹는지 보여드리겠습니다.

### 4.1. TensorFlow (Keras)의 '마법의 콜백(Callback)' 활용
Keras에서는 `EarlyStopping`과 `ModelCheckpoint`라는 비서(Callback) 두 명을 고용해서 `.fit()` 버튼을 누를 때 같이 넣어주기만 하면 끝납니다.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 모델 설계 (이전 단원의 Deep Net과 동일)
model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dropout(0.5), # 과적합 방지 1: 드롭아웃
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 2. 비서 1: EarlyStopping (더 이상 안 똑똑해지면 훈련 강제 종료)
# patience=3 : 3번의 에폭 동안 검증 오차(val_loss)가 안 떨어지면 가망이 없으니 스톱!
early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

# 3. 비서 2: ModelCheckpoint (가장 똑똑했던 순간의 뇌 상태를 디스크에 자동 저장)
# save_best_only=True : 오직 최고 신기록을 경신했을 때만 덮어쓰기 저장!
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)

# 4. 훈련 시작 (비서들을 callbacks 리스트에 담아 파견 보냄)
# epochs를 1000으로 엄청 크게 잡아도, early_stop 비서가 알아서 중간에 끊어줍니다!
model.fit(X_train, y_train, 
          validation_data=(X_valid, y_valid), 
          epochs=1000, 
          batch_size=128, 
          callbacks=[early_stop, checkpoint])

# 5. 훈련이 다 끝난 후, 디스크에 저장된 '최고 리즈시절'의 뇌 상태를 다시 불러오기
model = tf.keras.models.load_model('best_model.keras')
print("가장 똑똑했던 타이밍의 가중치로 복원 완료!")
```

### 4.2. PyTorch의 수동 얼리 스토핑 (직접 통제하기)
PyTorch는 편리한 비서가 없으므로, 우리가 훈련 루프(`for epoch in range(epochs):`) 안에 `if`문을 써서 직접 논리를 짜주어야 합니다. 이것이 파이토치의 묘미입니다!

```python
import torch

# 최고 기록을 저장할 변수 초기화 (처음엔 무한대로 세팅)
best_val_loss = float('inf')
patience = 3       # 3번 참아준다
trigger_times = 0  # 몇 번 참았는지 카운트

# 1. 에폭 반복 루프
for epoch in range(1000): # 1000번 돌린다고 세팅
    
    # ... (이전 단원에서 배운 model.train() 및 훈련 코드 5단계 생략) ...
    
    # 2. 검증 루프 진행
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.view(images.shape[0], -1))
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(test_loader)
    print(f'Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}')
    
    # 3. 얼리 스토핑 및 베스트 모델 저장 논리 (직접 코딩!)
    if avg_val_loss < best_val_loss:
        # 신기록 달성! 카운트 초기화하고 뇌 상태(가중치)를 파일로 저장
        best_val_loss = avg_val_loss
        trigger_times = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print("--> 신기록 갱신! 모델 저장됨.")
    else:
        # 신기록 달성 실패... 인내심 카운트 증가
        trigger_times += 1
        print(f"--> 신기록 실패. 인내심 카운트: {trigger_times}/{patience}")
        
        # 인내심이 한계에 달하면 강제 종료 (Break)
        if trigger_times >= patience:
            print("Early Stopping 발동! 훈련을 조기 종료합니다.")
            break

# 4. 훈련 완전 종료 후, 저장해둔 최고 신기록 뇌 상태 다시 불러오기
model.load_state_dict(torch.load('best_model.pth'))
print("가장 똑똑했던 타이밍의 가중치로 복원 완료!")
```

이처럼 실무에서는 어떤 라이브러리를 쓰든 **"훈련은 넉넉하게 걸어두되, 모델이 알아서 최고점을 찍고 스스로 훈련을 멈춘 뒤, 가장 좋았던 과거 상태로 돌아가도록(Early Stopping + Model Checkpoint)"** 세팅하는 것이 필수 중의 필수입니다! 

자, 이제 딥러닝 모델에 영혼을 불어넣는 튜닝 기술까지 모두 갖추셨습니다. 다음 마지막 단원에서는 나만의 딥러닝 프로젝트를 시작하기 위해 어떤 데이터셋을 구하고 어디서 더 공부해야 하는지 알아봅시다!
