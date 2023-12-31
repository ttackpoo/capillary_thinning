"""
MCPLExpt는 실험 데이터의 체계적인 관리와 분석을 위한 개발 툴입니다.

개발 배경
=========

연구실에서 진행되는 실험 결과는 대부분 따로 데이터베이스화 되지 않고 디렉토리 안에
파일로서 저장됩니다. 이 때, 한 디렉토리 안에 함께 저장되는 파일들은 일반적으로 같은
실험 변수를 공유하며, 때로는 한 실험을 위해 여러 파일이 생성되어 서로간의 참조가
필요하기도 합니다.

본 패키지는 이 점에 착안해, 실험 결과를 분석하는 코드를 작성하는 데 도움이 되기 위해
개발되었습니다.

디자인
======

본 문단은 :ref:`core` 모듈에 구현된 기본적인 디자인에 대해 설명합니다.
각 모듈의 클래스는 해당 모듈의 클래스를 상속하는 것을 기본으로 하나, 실제로는
상이한 방식으로 구현될 수 있습니다.

실험 데이터
-----------

본 패키지가 분석할 실험 데이터 디렉토리는 다음과 같이 실험 데이터가 저장된 파일들과
실험 변수가 저장된 하나의 파일로 구성되어야 합니다::

    실험 디렉토리  
    ├── 실험 파일 1  
    ├── 실험 파일 2  
    │   ...  
    └── 변수 파일  

데이터 분석 결과에 영향을 주는 모든 인자는 변수 파일을 통해 전달될 수 있어야 합니다.
변수 파일의 규칙 및 분석 방법은 자유롭게 구현될 수 있습니다.

클래스 구조
-----------

MCPLExpt는 두 종류의 추상적인 클래스를 기반으로 설계되었습니다.

- 데이터 디렉토리 클래스
    :class:`ExperimentSuite <mcplexpt.core.experiment.ExperimentSuite>` 클래스를
    상속받습니다. 파일 경로 및 파일명 인식에 대한 동작을 담당하며, ``get()``
    메소드를 통해 실험 파일들과 변수 파일 간의 연동이 구현됩니다.

- 데이터 파일 클래스
    :class:`Experiment <mcplexpt.core.experiment.Experiment>` 클래스를
    상속받습니다. 데이터 분석 알고리즘이 구현됩니다. 데이터 디렉토리 클래스의
    ``get()`` 메소드를 이용해 생성되어야 합니다.

예시
----

다양한 유체의 채널 흐름을 시각화하는 실험을 상정해 봅시다.
각 실험은 mp4 파일로 저장되고, 일련의 실험을 하기 전 기준을 잡기 위해 빈 채널을
촬영한 png 파일이 저장된다고 해 봅시다. 모든 실험에서 각 유체에 대한 점도, 밀도
등의 정보를 기록해야 하는데, 하나의 변수 파일에 이를 기록하고 저장하면 좋을
것입니다.

이 경우 실험 데이터 디렉토리는 다음과 같이 구성됩니다::

    데이터 디렉토리
    ├── reference.png
    ├── fluid1_expt1.mp4
    ├── fluid1_expt2.mp4
    ├── fluid2_expt1.mp4
    │   ...
    └── 변수 파일

이 때 분석 모듈은 다음과 같이 구현됩니다:

1. 데이터 파일 클래스 정의
    레퍼런스 이미지에 대한 클래스와, 실험 동영상에 대한 클래스가 따로 구현됩니다.
    각 실험 파일은 클래스의 인스턴스로 나타나며, 실험 변수는 어트리뷰트로 저장되고
    이를 이용한 영상 분석 알고리즘이 메소드로 구현됩니다.

2. 데이터 디렉토리 클래스 정의
    이 클래스에는 레퍼런스 파일과 실험 파일, 변수 파일을 구분하는 규칙이 등록됩니다.
    ``get()`` 메소드는 적합한 데이터 파일 인스턴스를 생성하고, 필요한 모든 변수를
    읽어와 인스턴스에 부여합니다. 따라서 데이터 분석 API는 이 클래스에 구현됩니다.

"""
