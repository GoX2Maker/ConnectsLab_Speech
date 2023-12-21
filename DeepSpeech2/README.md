# STT

Speech To Text를 진행한 과정을 설명하겠습니다.

## 1. Dataset 획득

본 STT 모델은 AI-Hub의 자유대화 음성 데이터로 학습을 진행했습니다.

## 2. 데이터 전처리

AI-Hub에서 얻은 데이터는 불필요한 데이터가 많습니다. 필요한 정보만 처리하기 위해
전처리 과정을 진행했습니다.

필요한 데이터는 다음과 같습니다.

- 파일 경로
- 텍스트 정보

이를 처리하기 위해 **prepcocessiong** 안에 있는 주피터 파일를 이용하면 됩니다.<br>(처리시 경로 및 파일명을 적절하게 수정해야합니다.)

## 3. 모델 Train

모델을 train하기 위해서 configs 폴더에 있는 trainConfigs.yaml을 수정이 필요합니다.

> **data/train_manifest** <br> 전처리한 csv 파일 경로를 입력합니다.

> **data/val_manifest** <br> 전처리한 csv 파일 경로를 입력합니다.

> **checkpointing/save_folder** <br> 모델을 저장할 위치를 지정합니다.

이외에 cuda, 하이퍼파라미터는 개인의 선택에 따라 정하면 됩니다.

터미널을 이용해서 다음 명령어를 입력해서 train을 진행합니다.

    $ python train.py위치 --config=trainConfigs.yaml위치

## 4. 모델 Eval

모델을 평가하기 위해서 configs 폴더에 있는 evalConfigs.yaml을 수정이 필요합니다.

> **data/test_manifest** <br> 전처리한 csv 파일 경로를 입력합니다.

> **model/savePath** <br> 평가할 모델 위치를 지정합니다.

이외에 cuda, 하이퍼파라미터는 train에 사용한 값을 그대로 사용합니다.

터미널을 이용해서 다음 명령어를 입력해서 평가를 진행합니다.

    $ python eval.py위치 --config=evalConfigs.yaml위치

## 5. 모델 실사용

학습한 모델이 잘 동작했을 때 이를 실제 음성파일을 이용해서 사용할 수 있습니다.

이를 이용하기 위해 configs 폴더에 있는 transcribeConfigs.yaml을 수정이 필요합니다.

> **data/audioFolder** <br> 음성파일이 있는 폴더경로를 입력합니다.

> **data/readCNT** <br> audioFolder에 있는 사용할 음성파일 수를 지정할 수 있습니다.<br>(단, 0으로 입력 시 모든 파일을 사용합니다.)

터미널을 이용해서 다음 명령어를 입력합니다.

    $ python transcribe.py위치 --config=transcribeConfigs.yaml위치
