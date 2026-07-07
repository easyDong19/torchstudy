# torchstudy

PyTorch를 공부하며 정리한 실습 노트북 모음입니다. 공식 PyTorch 튜토리얼을 따라 기초 개념(텐서, 자동미분, 데이터셋/데이터로더, 모델 구성, 최적화)을 익히고, 딥러닝 서적(회귀/분류/CNN/전이학습/RNN/LSTM)의 예제를 챕터별로 구현하며 학습한 내용을 기록하는 것이 목적입니다.

## 구성

### `tutorial/` — PyTorch 공식 튜토리얼 실습
PyTorch 공식 문서의 "Learn the Basics" 튜토리얼을 순서대로 따라간 노트북입니다.

- `Tesors.ipynb` — 텐서(Tensor) 생성, 속성, 인덱싱/슬라이싱, 연산(행렬곱, 원소곱), in-place 연산, GPU 이동
- `datasets_dataLoaders.ipynb` — `Dataset`/`DataLoader` 사용법, FashionMNIST 로드, 커스텀 `Dataset` 클래스 작성
- `build_the_newural_nn.ipynb` — `nn.Module`을 상속한 신경망 정의, `nn.Flatten`/`nn.Linear`/`nn.ReLU`/`nn.Sequential`, 파라미터 조회
- `Automatic_Differentiation.ipynb` — `requires_grad`와 `autograd`를 이용한 그래디언트 계산, `torch.no_grad()`
- `optimizer.ipynb` — 손실 함수(`CrossEntropyLoss`)와 옵티마이저(`SGD`) 정의, 학습/평가 루프(`train_loop`/`test_loop`) 작성
- `quickStart.ipynb` — 위 개념들을 종합한 FashionMNIST 분류 전체 파이프라인 (데이터 로드 → 모델 정의 → 학습 → 평가)

### `chapter03/` — 회귀·분류 기초
- `ex3_1.ipynb` — Tensor만으로 3차 다항식을 직접 구현해 경사하강법으로 sin 함수 근사 (autograd 없이 수동 학습)
- `ex3_2.ipynb` — 보스턴 주택 가격 데이터셋으로 `nn.Sequential` 기반 회귀 모델 학습 (배치/에포크/이터레이션 개념 정리)
- `ex3_3.ipynb` — MNIST 손글씨 데이터셋을 MLP로 분류, 학습된 가중치를 `MNIST.pth`로 저장하고 평가 정확도 측정

### `chapter04/` — CNN과 전이학습, 데이터 전처리
- `ex4_1.ipynb` — `nn.Module`과 `nn.Sequential`의 차이, `BasicBlock`을 쌓아 CNN을 직접 구현하고 CIFAR-10 분류
- `ex4_2.ipynb` — 사전학습된 VGG16을 불러와 CIFAR-10에 맞게 파인튜닝하는 전이학습 실습
- `preprocessing.ipynb` — 이미지 데이터 증강(크롭, 좌우대칭), 정규화, CIFAR-10 평균/표준편차 계산 등 전처리 기법 정리

### `chapter06/` — RNN
- `ex6_1.ipynb` — 넷플릭스 주가 데이터(`CH06.csv`)를 최소-최대 정규화 후, 커스텀 `Dataset`과 `nn.RNN`으로 시계열 예측 모델 학습

### `chapter10/` — LSTM을 이용한 텍스트 생성
- `ex10_1.ipynb` — 뉴욕타임즈 기사 제목 데이터(`ArticlesFeb2017.csv`)로 단어 임베딩 + LSTM 언어모델을 학습해 다음 단어를 예측하는 텍스트 생성기 구현

## 참고
- 각 노트북은 독립적으로 실행 가능하며, 학습된 모델 가중치(`.pth`)와 데이터셋은 해당 챕터 폴더에 함께 저장됩니다.
- 주요 사용 라이브러리: `torch`, `torchvision`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `tqdm`
