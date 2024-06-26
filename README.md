# Drowsiness-Detection-based-on-Eye-Recognition-using-OpenCV
이 프로젝트는 OpenCV를 활용하여 운전자의 눈 깜박임을 감지하고, 일정 시간 동안 눈을 감고 있을 때 경고 메시지를 출력하는 졸음 감지 시스템.

## 프로젝트 개요
졸음운전은 전 세계적으로 심각한 교통사고의 주요 원인 중 하나로, 인명피해와 경제적 손실을 초래하고 있습니다. 특히, 장시간 운전이나 야간 운전 중에 졸음운전의 위험이 크게 증가합니다. 졸음운전은 운전자의 반응 시간을 늦추고, 집중력을 저하시켜 사고 발생 가능성을 높입니다. 
경찰청 교통사고 통계에 따르면 2023년에는 사고 2,016건, 사망 48명이고, 2022년은 사고 1,849건 , 사망 55건 등  지난 5년간(2019년~2023년) 졸음운전으로 인한 교통사고는 총 10,765건으로 하루 평균 5.9건이 발생하였습니다.

이 프로젝트는 운전자가 눈을 2초 이상 감고 있는 경우, "눈을 감고 있습니다"라는 텍스트를 화면에 출력하여 경고하는 시스템을 구현.

- **프로그래밍 언어**: Python (version : 3.11)
- **라이브러리**: OpenCV

## 기능

- OpenCV를 사용한 실시간 눈 인식
- 눈을 2초 이상 감고 있을 때 경고 메시지 출력

## 사용법
웹캠을 통해 실시간 영상을 캡처합니다. 
눈 인식을 시작합니다.
눈을 2초 이상 감고 있을 경우, 화면에 경고 메시지가 출력됩니다.
