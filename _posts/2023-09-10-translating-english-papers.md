---
layout: post
title: 영어 논문 번역 Tips
date: 2023-09-10 16:13 +0800
last_modified_at: 2023-09-10 17:50 +0800
tags: [paper, translate]
toc: true
---

기계학습을 공부하면서 언어의 장벽에 부딪히는 것은 흔한 일입니다. 대부분의 유용한 논문들이 영어로 쓰여 있기 때문이죠.

무료로 읽을 수 있는 논문들은 주로 [arXiv.org e-Print archive](https://arxiv.org/)에 공개되어 있습니다. 예를 들어, 아래와 같은 논문들을 무료로 읽을 수 있습니다.

<물체 탐지(object detection)>

[[1512.02325] SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
[[1612.08242] YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)

<word2vec, doc2vec>

[[1301.3781] Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
[[1405.4053] Distributed Representations of Sentences and Documents](https://arxiv.org/abs/1405.4053)

영어 수준이 낮아도 Google 번역의 딥러닝 기반 정확도 향상으로 인해 거의 독해에 지장이 없는 수준으로 번역할 수 있게 되었습니다.

새로운 Google 번역(GNMT)은 모든 Google 서비스에서 지원하는 것은 아니며, 예를 들어 Google 문서 번역 기능에는 오래된 엔진이 사용되고 있는 것 같습니다.

최근 Cloud Translation API가 GNMT를 지원했기 때문에 이를 사용하면 어느 정도 번역 작업이 자동화될 수 있을 것 같습니다만, 여기서는 수작업으로 브라우저의 Google 번역을 사용하여 번역할 때 작업 효율을 높이는 방법을 소개하겠습니다.

### 번역 절차

arXiv.org 등에서 구할 수 있는 논문은 대부분 PDF 형식인 경우가 많습니다.

PDF에서 영문을 복사하여 브라우저의 Google 번역에 붙여 넣으면 아래와 같은 문제가 발생합니다.

- 줄 끝에서 줄 바꿈이 발생
- 줄 끝에서 단어가 하이픈으로 되어 있으면 오역
- et al.(논문 저자의 기타 의미)이 마침표로 해석
- 페이지를 가로지르는 글 사이에 주석이나 바닥글이 포함

이러한 문제들은 수작업으로 수정해야 합니다.

복사한 텍스트를 텍스트 에디터에 붙여 가공해도 되지만, Google 문서에서 PDF를 Google 문서로 변환하면 줄 바꿈이 되지 않고 단락이 하나로 이어집니다.

또한 장과 절이 굵게 표시되어 텍스트 에디터에서 일괄 변환할 때보다 구조를 파악하기 쉽습니다.

PDF를 Google 문서로 변환하려면 Google 드라이브에 PDF 파일을 드롭하고, PDF 파일을 마우스 오른쪽 버튼으로 클릭한 후 앱에서 열기 → Google 문서로 변환을 선택하면 됩니다.

Google 문서에서 열어서 바꾸기 기능을 이용해 위의 하이픈 등을 없애는 작업을 한 후 Google 번역기에 단락 단위로 복사해서 붙여넣고 번역문을 단락 아래에 붙여 넣으면 나중에 영문과 번역문이 대비되어 보기 편합니다.

단, 브라우저의 Google 번역기에서 직접 보는 것이 단어를 선택해 의미를 확인하거나 발음을 확인할 수 있기 때문에 병행하면서 해석하는 것을 추천합니다.

영문 중 인용문헌을 ()로 표기하면 오역하는 경우가 많으므로 붙여넣기 시 삭제하는 것이 좋습니다.

또한, 수식이나 표는 깨져서 변환되므로 Acrobat의 스냅샷 기능으로 그림으로 복사하여 붙여 넣는 것이 좋습니다.

그림도 삭제되므로 PDF에서 복사하여 그림으로 붙여 넣으면 좋습니다. 이렇게 하면 번역과정이 더욱 효율적이게 됩니다.