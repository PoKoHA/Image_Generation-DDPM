# Denoising Diffusion Probabilistic models

1. **Forward(Diffusion) Process**: original Image에 조금씩 노이즈(Gaussian Noise)를 입혀 원래 Image와 Independent한 Noise Image 생성.
2. **Reverse Process**: 완전히 Gaussian Noise가 된 Image에 조금씩 Denoising하여 Original과 다른 새로운 Image 생성.

**<Forward Process를 학습하는 것이 아닌 Reverse Process를 학습>**

![Screenshot from 2022-01-17 14-05-16](https://user-images.githubusercontent.com/76771847/149711049-87030324-df3a-4e35-bb62-d54c89f52355.png)

- **Flow**

![Screenshot from 2022-01-17 14-06-47](https://user-images.githubusercontent.com/76771847/149711143-d2eaa2c2-4b68-42b7-b356-d00b9b6844d8.png)

# Forward Process
원본 이미지에 가우시안 노이즈를 입히는 과정

![1](https://user-images.githubusercontent.com/76771847/149711547-39edae59-79cb-4079-8da3-1fe4e623144a.png)


> **What distinguishes diffusion models from other types of latent variable models is that the approximate
posterior q(x1:T |x0), called the forward process or diffusion process, is fixed to a Markov chain that
gradually adds Gaussian noise to the data according to a variance schedule β1, . . . , βT**
- q(x1:T|X0): 노이즈를 Schedule에 따라 점진적으로 추가하는 **Markov Chain**으로 설정.

# Reverse Process
가우시안 노이즈가 된 이미지를 점진적으로 Denoising 하는 과정

![2](https://user-images.githubusercontent.com/76771847/149712807-108fda64-db60-464a-b774-c52963e4b4ba.png)

> **The joint distribution
pθ(x0:T ) is called the reverse process, and it is defined as a Markov chain with learned Gaussian
transitions starting at p(xT ) = N (xT ; 0, I)**


# Reference

- [**논문 설명 1**](https://developers-shack.tistory.com/8)
- [**논문 설명 2**](https://wain-together.tistory.com/9)
- [**Official Code**](https://github.com/hojonathanho/diffusion)
- [**참고 Code**](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/8af24da2dd39a9a87482a4d18c2dc829bbd3fd47/labml_nn/diffusion/ddpm/__init__.py#L172)

