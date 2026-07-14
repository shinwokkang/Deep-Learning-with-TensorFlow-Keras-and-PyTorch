# Lesson 5.2: 파이토치(PyTorch) 딥러닝 실전 입문 - Part 1

---

## 1. 파이토치(PyTorch)의 세계로 오신 것을 환영합니다!

이전 단원(5.1)에서 우리는 텐서플로우(Keras)가 '자동화된 레고 조립'이라면, 파이토치는 '수동 배관 공사'와 같다고 비유했습니다. 이제 그 배관 공사를 우리 손으로 직접 해볼 차례입니다. 

초보자분들은 처음 파이토치 코드를 보면 "Keras에서는 3줄이면 끝났는데, 왜 이렇게 코드가 길고 복잡해?"라며 당황할 수 있습니다. 하지만 걱정하지 마세요. 코드가 길어진 이유는 **"마법 상자(Black Box) 안에 숨겨져 있던 딥러닝의 진짜 수학적 흐름을 우리 눈앞에 펼쳐놓았기 때문"**입니다. 

파이토치를 배우는 과정은 단순히 도구를 배우는 것을 넘어, **딥러닝이 내부적으로 어떻게 굴러가는지 그 심장 박동을 직접 만져보는 가장 훌륭한 해부학 시간**이 될 것입니다.

우리의 첫 번째 목표는 아주 단순합니다. 우리가 텐서플로우에서 맨 처음 만들었던 **'얕은 신경망(Shallow Neural Network, 784 -> 64 -> 10)'**을 파이토치 문법으로 완벽하게 똑같이 번역(Replication)해 보는 것입니다.

---

## 2. Keras와 PyTorch의 훈련 철학 비교 (마법 버튼 vs 수동 조종석)

코드를 보기 전에, 아래의 이미지를 통해 Keras와 PyTorch가 훈련(Training)을 대하는 철학적 차이를 먼저 이해해 봅시다.

![Keras vs PyTorch 훈련 철학 비교](/Users/shinwookkang/.gemini/antigravity/brain/cb5306ae-e4e0-4381-a3ab-20a49944023c/keras_vs_pytorch_training_loop_1784003863426.jpg)

*   **왼쪽 (Keras)**: 모든 복잡한 톱니바퀴가 `.fit()`이라는 거대한 마법 버튼 하나 안에 숨겨져 있습니다. 사용자는 버튼만 누르면 됩니다.
*   **오른쪽 (PyTorch)**: 거대한 기계의 덮개를 열어젖혔습니다. '에폭(Epoch) 반복', '배치(Batch) 반복', '순전파(Forward)', '오차 계산(Loss)', '역전파(Backward)', '가중치 업데이트(Optimizer)'라는 모든 톱니바퀴를 사용자가 직접 손으로 조작해야 합니다. 

이제 이 수동 조종석에 앉아, 첫 번째 스위치부터 켜보겠습니다.

---

## 3. 데이터 준비하기 (Data Loading & DataLoader)

Keras에서는 데이터를 불러오는 것이 아주 쉬웠습니다. `mnist.load_data()` 한 줄이면 끝났죠. 하지만 파이토치에서는 조금 더 체계적이고 세밀한 설정이 필요합니다.

### 3.1. 라이브러리 임포트
먼저 파이토치와 관련된 라이브러리를 불러옵니다.
```python
import torch
from torchvision import datasets, transforms
# torchsummary는 Keras의 model.summary()와 같은 기능을 제공하는 유용한 외부 라이브러리입니다.
from torchsummary import summary 
import matplotlib.pyplot as plt
```

### 3.2. 데이터 다운로드 및 텐서 변환 (Transform)
파이토치에서 데이터를 불러올 때는 `torchvision.datasets`를 사용합니다. 이때 Keras에는 없던 몇 가지 특이한 인자들이 등장합니다.

```python
# 데이터를 변환할 규칙을 정합니다. (0~255 픽셀값을 0~1 사이의 실수(Float)로 자동 스케일링해주는 마법의 함수)
transform = transforms.ToTensor()

# 훈련 데이터 불러오기
train_data = datasets.MNIST(root='./data',        # 데이터를 저장할 폴더 경로 (Keras엔 없던 개념)
                            train=True,           # 훈련용 데이터인가? True
                            download=True,        # 폴더에 데이터가 없으면 다운로드할 것인가? True
                            transform=transform)  # 아까 만든 스케일링 규칙 적용!

# 테스트 데이터 불러오기
test_data = datasets.MNIST(root='./data', 
                           train=False,          # 테스트용이므로 False
                           download=True, 
                           transform=transform)
```

**🤔 중요한 질문: "데이터를 불러오자마자 바로 0~1로 스케일링이 되나요?"**
아닙니다! 파이토치는 **'게으른 실행(Lazy Execution)'** 방식을 씁니다. 지금 규칙(Transform)을 정해놓긴 했지만, 실제로 데이터가 메모리에 올라와서 0~1로 바뀌는 순간은 나중에 모델에 데이터를 쑤셔 넣기 직전입니다. 효율적인 메모리 관리를 위해서죠.

### 3.3. 데이터 로더 (DataLoader) 생성 - 미니배치 공장 만들기
Keras에서는 `.fit(batch_size=128)`처럼 괄호 안에 적어주면 알아서 배치를 쪼개주었습니다. 파이토치에서는 데이터를 쪼개고 섞어주는 공장장인 **`DataLoader`**를 따로 만들어야 합니다.

```python
# 훈련 데이터 로더 (미니배치 128개씩 쪼개고, 마구 섞어라(Shuffle=True))
# Stochastic Gradient Descent를 위해 훈련 데이터는 무조건 섞어야 합니다.
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)

# 테스트 데이터 로더 (섞을 필요 없음)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)
```
이제 `train_loader`에게 "데이터 하나만 줘봐!"라고 하면, 알아서 128장의 이미지를 0~1로 예쁘게 스케일링해서 묶음으로 던져줍니다.

---

## 4. 데이터 모양 바꾸기 (Flattening의 비밀)

우리의 얕은 신경망은 28x28 형태의 2차원 이미지를 소화하지 못합니다. 일렬로 쫙 펴서(Flatten) 784개의 1차원 선으로 만들어야 합니다. Keras에서는 `Flatten()` 레이어를 추가하면 끝이었지만, 파이토치 텐서에서는 **`view()`** 또는 **`reshape()`** 함수를 사용해 직접 모양을 찌그러뜨립니다.

```python
# train_loader에서 128장짜리 묶음을 하나 뽑아왔다고 가정합시다.
# x_sample의 현재 모양(Shape)은 [128, 1, 28, 28] 입니다. (128장, 흑백(1), 가로 28, 세로 28)

# 이걸 [128, 784] 모양으로 찌그러뜨리고 싶습니다.
flattened_x = x_sample.view(x_sample.shape[0], -1) 
# 해석: 첫 번째 차원(128)은 그대로 유지하고, 나머지 모든 차원(1x28x28)은 알아서(-1) 하나로 곱해서 뭉쳐라!
# 결과: [128, 784]
```
여기서 `-1`은 파이썬에게 "네가 알아서 남은 픽셀 수를 계산해서 채워 넣어!"라고 시키는 아주 유용한 꼼수입니다. 데이터 60,000장을 128로 나누면 맨 마지막 자투리 배치는 128장이 안 되고 96장이 남게 되는데, 이때 하드코딩으로 `view(128, 784)`라고 적어두면 마지막 배치에서 에러가 납니다. 하지만 `-1`을 쓰면 자투리가 96장일 때는 유연하게 `[96, 784]`로 알아서 계산해 줍니다!

---

## 5. 파이토치로 신경망 뼈대 조립하기 (수동 배관 공사)

이제 드디어 파이토치 문법으로 우리가 알던 얕은 신경망을 조립해 보겠습니다. 파이토치에서는 `nn.Module`이라는 클래스를 상속받아 나만의 모델 공장을 차려야 합니다.

```python
import torch.nn as nn
import torch.nn.functional as F

# Keras의 Sequential처럼 부품들을 담아둘 나만의 모델 클래스 생성
class ShallowNet_PyTorch(nn.Module):
    def __init__(self):
        super(ShallowNet_PyTorch, self).__init__()
        
        # [부품 구매 단계] Keras의 Dense 레이어는 PyTorch에서 Linear 레이어라고 부릅니다.
        # 1. 은닉층: 784개의 픽셀을 입력받아 64개의 뉴런으로 보낸다.
        self.fc1 = nn.Linear(in_features=784, out_features=64)
        
        # 2. 출력층: 64개를 입력받아 최종 10개(0~9 숫자)의 뉴런으로 보낸다.
        self.fc2 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # [부품 조립 단계] 데이터(x)가 실제로 흘러가는 배관을 연결합니다.
        
        # 1. 28x28 이미지를 784로 쫙 폅니다.
        x = x.view(x.shape[0], -1) 
        
        # 2. 은닉층(fc1)을 통과시키고, Sigmoid 활성화 함수 마법을 씌웁니다.
        x = self.fc1(x)
        x = torch.sigmoid(x) # (참고: 실무에선 ReLU를 쓰지만, 영상 복습을 위해 Sigmoid 사용)
        
        # 3. 출력층(fc2)을 통과시킵니다.
        x = self.fc2(x)
        
        return x

# 모델 완성!
model = ShallowNet_PyTorch()
summary(model, input_size=(1, 28, 28)) # Keras처럼 모델 요약본 보기
```

**💡 [매우 중요] Keras와의 결정적 차이 2가지**
1.  **활성화 함수(Activation)의 위치**: Keras는 `Dense(64, activation='sigmoid')`처럼 부품 안에 마법을 포함시켜서 샀습니다. 반면 파이토치는 부품(`nn.Linear`)을 통과한 직후에 배관(`forward`)에서 수동으로 마법(`torch.sigmoid`)을 뿌려줍니다.
2.  **출력층의 Softmax가 없다?!**: Keras에서는 맨 마지막에 `Dense(10, activation='softmax')`를 썼습니다. 하지만 위 파이토치 코드에는 Softmax가 없습니다! 그 이유는 **파이토치의 손실 함수(Loss Function)가 내부에 Softmax를 이미 품고 있기 때문**입니다. 모델 끝에 Softmax를 중복해서 달면 계산이 꼬이게 됩니다. (이것이 파이토치 초보자들이 가장 많이 하는 실수입니다!)

---

## 6. 하이퍼파라미터 세팅과 그 악명 높은 '훈련 루프(Training Loop)' 짜기

이제 모델을 훈련시킬 준비가 끝났습니다. 오차를 계산할 채점관(Loss)과 가중치를 업데이트할 최적화 요정(Optimizer)을 고용합니다.

```python
import torch.optim as optim

# 채점관 고용: CrossEntropyLoss (내부에 Softmax 마법이 탑재되어 있음!)
criterion = nn.CrossEntropyLoss()

# 최적화 요정 고용: 확률적 경사 하강법(SGD). 
# 요정에게 "이 모델(model.parameters())의 가중치를 학습률 0.1로 고쳐줘!"라고 지시서(명함)를 건넵니다.
optimizer = optim.SGD(model.parameters(), lr=0.1)
```

### 🥵 딥러닝 해부학의 꽃: 수동 훈련 루프 (Manual Training Loop)
이제 Keras의 `.fit()` 버튼 하나에 숨겨져 있던 그 길고 긴 코드를 우리 손으로 직접 타이핑할 시간입니다. (천천히 주석을 따라 읽어보세요. 딥러닝의 원리가 머릿속에 완벽히 그려질 것입니다.)

```python
epochs = 5

# 1. 에폭 톱니바퀴 (전체 데이터를 몇 번 반복할 것인가?)
for epoch in range(epochs):
    
    # 2. 미니배치 톱니바퀴 (데이터 로더에서 128장씩 묶음을 꺼내옴)
    for batch_idx, (images, labels) in enumerate(train_loader):
        
        # ----------------------------------------------------
        # [딥러닝의 심장 박동 5단계]
        # ----------------------------------------------------
        
        # Step 1. 최적화 요정의 기억 지우기 (초기화)
        # 파이토치는 특이하게도 이전 배치의 기울기(Gradient) 쓰레기가 남아있습니다. 
        # 그래서 매번 새로운 배치를 학습하기 전에 쓰레기통을 비워줘야 합니다.
        optimizer.zero_grad() 
        
        # Step 2. 순전파 (Forward Pass)
        # 128장의 이미지를 모델에 통과시켜 예측값(outputs)을 얻어냅니다.
        outputs = model(images) 
        
        # Step 3. 오차 계산 (Calculate Loss)
        # 채점관이 모델의 예측(outputs)과 실제 정답(labels)을 비교하여 오차 점수를 매깁니다.
        loss = criterion(outputs, labels)
        
        # Step 4. 역전파 (Backward Pass)
        # 오차를 바탕으로 각 가중치(w, b)가 얼마나 잘못되었는지 기울기(미분값)를 계산하여 거꾸로 전달합니다.
        loss.backward()
        
        # Step 5. 가중치 업데이트 (Optimizer Step)
        # 최적화 요정이 계산된 기울기 방향으로 가중치를 살짝 이동(업데이트)시킵니다.
        optimizer.step()
        
        # ----------------------------------------------------
        
        # 로그 출력 (선택 사항: 100번째 배치마다 현재 훈련 상황을 화면에 보여줌)
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
```

정말 길고 복잡해 보이지만, 사실 딥러닝의 핵심은 저 **[심장 박동 5단계]**가 전부입니다.
`zero_grad() -> forward() -> loss() -> backward() -> step()`
파이토치를 쓰는 모든 전 세계의 AI 연구자들은 숨을 쉬듯 이 5단계를 무한히 반복하며 코드를 짭니다.

### 🎯 정확도(Accuracy) 계산의 수동화
Keras에서는 `metrics=['accuracy']` 한 줄이면 알아서 정확도를 퍼센트로 보여줬지만, 파이토치는 이마저도 손으로 짜야 합니다. (아래 코드는 이해만 하고 넘어가셔도 좋습니다.)

```python
# 정확도 계산 함수 예시
def calculate_accuracy(outputs, labels):
    # 가장 확률이 높은 숫자(0~9)의 인덱스를 찾아냅니다.
    _, predicted = torch.max(outputs, 1) 
    # 정답과 예측이 똑같은 개수를 센 다음, 전체 개수로 나누어 퍼센트를 만듭니다.
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0) * 100
    return accuracy
```

---

## 7. 결론: 왜 이런 사서 고생을 해야 할까요?

수고하셨습니다! 여러분은 방금 가장 근본적인 형태의 파이토치 신경망 훈련 루프를 완성했습니다. Keras에 비하면 타이핑할 코드량이 5배는 많아진 것 같은데, 굳이 왜 파이토치를 쓸까요?

바로 **"자유도"** 때문입니다. 
만약 우리가 "에폭이 홀수일 때는 A 방식으로 학습하고, 에폭이 짝수일 때는 B 방식으로 학습하며, 오차가 특정 수준 이하로 떨어지면 특정 가중치에만 패널티를 주는 이상한 훈련을 시키고 싶어!"라고 한다면 Keras의 꽉 막힌 `.fit()` 버튼 안에서는 구현하기가 하늘의 별 따기입니다. 

하지만 파이토치는 저 `for`문 안에 우리가 원하는 파이썬 `if`문이나 수학 공식을 내 마음대로 끼워 넣기만 하면 즉각적으로 실험이 가능합니다. 이것이 바로 파이토치가 전 세계 논문과 최전선 연구소(Research)의 표준이 된 결정적인 이유입니다.

이제 파이토치의 뼈대를 잡았으니, 다음 강의 **Part 2**에서는 이를 더 심화시켜 실무에서 사용하는 형태로 다듬어 보겠습니다!
