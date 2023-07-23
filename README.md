# Code Availability





## Over view
<img src="/code-availability/mcplexpt/Figure/Readme_figure1.png" width="1100" />

* **Step1. Data Collection**: Fluid thinning 영상 이미지를 측정하고, 로딩해서 필요한 이미지들만 추출하는 단계
* **Step2. Data Processing**: Whitening, Cropping, Stacking 과정을 통해서 Merge Image를 만들고, Data Augmentation 활용하여 Data 증폭하는 단계
* **Step3. Model Training**: PCA 와 KNN method 활용하여 Model을 유체 농도 예측 모델을 트레이닝하는 단계
* **Step4. Model Validation**: MSE 방법을 활용해서 PCs,k,Weigtht,Frame 등의 Hyper parameter를 결정하는 단계
* **Step5. Model Testing**: 최종적으로 Test Data를 활용하여 유체 농도를 예측하는 단계


## Step1. Data Collection

첫 번째 단계는 우리 페이퍼(연동)의 Introduction에서와 같이 DoS-CaBER를 사용하요 확보된 영상 데이터를 불러오는 단계이다. 해당 단계는 dos.py(강조) 파일을 통해 실행된다. 영상은 DoSCaBERExperiment.get_nth_image를 통해 frame 순서대로 불러와진다. DoSCaBERExperiment.capbridge_start와 DoSCaBERExperiment.capbridge_broken method를 통해 fluid thinning이 발생되는 frame과 fluid pinch-off가 발생하는 frame을 찾아낸다. 우리는 본 논문에서 그 2개의 frame과 그 사이의 frame들을 사용한다.  [paper](https://epicgit.snu.ac.kr/ttackpool/paper_minhyuckim_eigen_thinning/-/blob/main/sn-article.pdf)

## Step2. Data Processing
두 번째 단계는 앞서 확보한 이미지 데이터들을 모델 구축에 필요한 데이터 형태로 가공하는 단계이다. 본 논문에서는 크게 2개의 데이터 가공방법이 활용되었따. 첫 번째 가공방법은 Merge된 Fluid thinning 이미지를 얻는 방법이다. 이를 위해서는 이미지 데이터의 Whitening, Cropping, Stacking이 필요하다. 이는 dos.py(강조) 파일을 통해 실행된다. DoSCaBERExperiment.Image_storage method를 활용한다. 해당 method 에서는 OpenCV 라이브러리를 사용하여 이미지를 Whitening 한다. 또한 이미지의 형상적 특징을 기반으로 Cropping을 하여, 불필요한 이미지 데이터를 삭제하고 이미지의 Centering을 맞춘다. 각 Frame은 White pixel은 255/Number of total frames로 만들어지고 모든 frame들을 하나의 이미지로 합친다. 이과정을 통해 Fluid thinning 영상은 1개의 이미지로 통합된다. 마지막으로 필요에 따라 Pinch-off되는 시점의 이미지들에는 Weight가 주어진다. 이러한 과정의 필요성은 본 논문(연동)의 Results and discussion 과 Fig.8과 Fig.9에서 설명되어진다. 두 번째 가공방법은 Image data들을 Augmentation 하는 방법이다. 이는 [PCA]((https://epicgit.snu.ac.kr/ttackpool/code-availability.git)) 디렉토리의 `PCA.py` 파일을 통해 실행되어진다. `PCA.pcaclass.augmentation` 를 통해 구현된다. 이 방법은 본 논문에서 PCA를 통해 구현한 독자적인 Fluid thinning의 Augmentation 방법이다. 각 PC에 해당하는 PCA score값에 random ratio를 곱하여 변환을 주고 이를 reconstruction 해서 PCs가 조금 씩 변환된 Image들을 만들어 낸다. Image augmentation은 모델의 정확도 향상을 만든다.
## Step3. Model Training
세 번째 단계는 혼합 유체의 농도를 예측하는 Machine learning model을 트레이닝하는 단계이다. 해당 단계는 [PCA]((https://epicgit.snu.ac.kr/ttackpool/code-availability.git)) 디렉토리의 `PCA.py`파일을 통해 실행된다. `PCA.pcaclass.eigen(수정필요)`을 통해 농도 예측 모델은 만들어진다. scikit-learn 라이브러리가 주요하게 사용되었다. `sklearn.decomposition.PCA`을 통해 차원축소 및 노이즈 제거 방법 중 한가지인 PCA를 실행하였다. PCA가 적용된 데이터들과 `sklearn.decomposition와 sklearn.neighbors.KNeighborsClassifier`을 사용하여  `KNN 알고리즘` 기반의 유체농도 예측 모델을 만들었다. 

## Step4. Model Validation
네 번째 단계는 모델의 정확도 향상을 위한 Hyper parameter 결정을 위한 Validation 단계이다. `PCA.pcaclass.eingen_validation(수정필요)`을 사용하여 진행된다. 우리의 모델에서 농도를 예측하는 계산 방식은 논문의 ~~에 부분에서 설명된다. `knn_proba`를 활용하여 validation data의 각 class에 해당하는 probability가 계산된다. probability를 기반으로 validation data의 농도가 예측된다. 예측값과 label값의 차이를 통해 `MSE(Mean Squre Error)`를 구한다. 이 때, KNN의 class 결정 parameter인 k 값은 6으로 설정되었다. 이 값은 유체를 예측하는 모델의 특성과 explainable한 PCs의 Eigen-thinning 이미지를 기반으로 설정되었다. 다음으로 PCA를 통해 얻어진 PCs의 갯수와 weight를 주는 frame의 수와 Weight의 정도에 대해 iterative한 계산을 시행한다. 최종적으로, iterative한 계산의 결과에서 MSE가 최소가 되는 PCs의 갯수, Frame의 수, Weight값이 결정된다. 본 논문에서는 유체의 농도를 예측하기 위해 이와 같은 방식으로 MSE를 구하였지만, 이는 사용자의 목적에 따라 예측값을 계산하는 과정이 달라질 수 있으며 이에 따라 Model 정확도를 향상시키는 Hyper parameter 역시 달라질 수 있다.

## Step5. Model Testing
다섯 번째 단계는 모델을 테스트하는 단계로, 유체의 농도를 예측하는 단계이다. `PCA.pcaclass.Ratio_plot`을 활용한다. 이 때는 결정된 PCs가 parameter로 입력되어야 한다. 결과적으로 실제 농도와 예측된 농도가 Bar graph image로 표현된다.
네 번째 단계는 모델의 정확도 향상을 위한 Hyper parameter 결정을 위한 Validation 단계이다. `PCA.pcaclass.eing_validation(수정필요)`을 사용하여 진행된다. 이 때, KNN의 class 결정 parameter인 k 값은 1~20까지 구간에서, 동시에 PCA의 주성분의 갯수는 5~55까지 5의 간격으로 반복되어 계산된다. 농도를 예측하는 계산 방식은 논문의 ~~에 부분에서 설명된다. `knn_proba`를 활용하여 validation data의 각 class에 해당하는 probability가 계산된다. probability를 기반으로 validation data의 농도가 예측된다. 예측값과 label값의 차이를 통해 `MSE(Mean Squre Error)`를 구한다. MSE가 최소가 되는 k값과 PCs를 결정한다. 본 논문에서는 유체의 농도를 예측하기 위해 이와 같은 방식으로 MSE를 구하였지만, 이는 사용자의 목적에 따라 예측값을 계산하는 과정이 달라질 수 있으며 이에 따라 Model 정확도를 향상시키는 Hyper parameter는 달라진다.

## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Images](https://epicgit.snu.ac.kr/ttackpool/code-availability/-/tree/main/mcplexpt/caber)

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://epicgit.snu.ac.kr/ttackpool/code-availability.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://epicgit.snu.ac.kr/ttackpool/code-availability/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Automatically merge when pipeline succeeds](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing(SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thank you to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README
Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
