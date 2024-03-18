# VISION AI

- **Computer Vision as a Service (CVaaS**: Many companies are offering cloud-based computer vision services that allow businesses to integrate advanced image recognition capabilities into their applications without needing in-depth AI expertise. These services include image classification, object detection, and facial recognition, among others.

- **Automated Quality Inspection**: In manufacturing, AI vision is being used for quality control processes. These systems can detect defects or irregularities in products at a speed and accuracy level that far exceeds human capabilities. This not only reduces costs but also improves product quality.

- **Retail Analytics**: AI vision technologies are being used in the retail sector for customer behavior analysis, inventory management, and enhancing the shopping experience through personalized recommendations. This includes tracking customer movements within stores to analyze traffic patterns and product interactions.

- **Healthcare Diagnostics**: AI vision is revolutionizing healthcare by assisting in the diagnosis of diseases from medical imagery such as X-rays, MRIs, and CT scans with greater accuracy and speed. This includes detecting cancers at earlier stages, thus significantly improving patient outcomes.

- **Augmented Reality (AR** and Virtual Reality (VR: With the integration of AI vision, AR and VR technologies are becoming more interactive and immersive. This is used in various applications, from gaming and entertainment to education and training simulations.

- **Generative AI for Content Creation**: Beyond analysis, AI vision is also being used to generate visual content. This includes creating realistic images, videos, and simulations for entertainment, marketing, and training purposes. Generative Adversarial Networks (GANs are a key technology here.

- **Edge AI in Vision**: There's a growing trend towards processing AI tasks at the edge, i.e., on local devices, rather than in the cloud. This reduces latency, increases privacy, and lowers bandwidth requirements for applications like real-time video analytics and IoT devices.

- **Autonomous Vehicles**: AI vision is a critical component of autonomous vehicle technology, enabling vehicles to interpret and understand the environment around them to navigate safely. This includes object and pedestrian detection, traffic sign recognition, and real-time decision-making.

- **Ethical and Responsible AI**: As AI vision technologies become more pervasive, there's an increased focus on ethical considerations, including privacy, bias, and accountability. Consultancy companies are now offering services to help businesses deploy AI vision solutions in an ethical and responsible manner.

- **Custom AI Models**: While pre-trained models offer a great starting point, there's a trend towards custom AI models tailored to specific business needs and data. Consultancies are helping companies develop these custom models to maximize performance and relevance to their unique challenges.





# PRETRAINED MODELS

## MODELS ON HUGGINGFACE
- Models on Hugging Face include image classification, object detection, semantic segmentation, and more. 
- Some of the popular architectures available include Convolutional Neural Networks (CNNs like ResNet, EfficientNet, and Vision Transformers (ViT.

### Typical Usage of Pretrained Models
- **Image Classification**: One of the most common uses of pretrained models is to categorize images into predefined classes. Models trained on large datasets like ImageNet are often used as starting points for specific image classification tasks.
- **Object Detection**: Pretrained models are also used to identify and locate objects within images. This is crucial for applications like surveillance, vehicle navigation, and retail analytics.
- **Semantic Segmentation**: In tasks where understanding the context of each pixel in an image is necessary, semantic segmentation models come into play. They are widely used in medical imaging, autonomous vehicles, and land use/land cover mapping in satellite imagery.
- **Transfer Learning**: A significant use case for pretrained models is transfer learning, where a model trained on one task is fine-tuned for another, related task. This approach allows for leveraging the knowledge gained from large, diverse datasets, even when the target task has limited data available.
- **Feature Extraction**: Pretrained models are also used as feature extractors. In this scenario, the output of one of the intermediate layers of the model is used as a compact representation of the input image. These features can then be used for various tasks, such as similarity search or clustering.





## MODELS

### STABLE DIFFUSION
What’s the advantage of Stable Diffusion? There are similar text-to-image generation services like DALLE and MidJourney. Why Stable Diffusion? The advantages of Stable Diffusion AI are:
- Open-source: Many enthusiasts have created free tools and models.
- Designed for low-power computers: It’s free or cheap to run.


[Stable Diffusion Models: a beginner’s guide](https://stable-diffusion-art.com/models/) (jest info jak działać w AUTOMATIC1111 GUI)

TODO: sprawdz najlepsze modele runwayml, openai, stabilityai, Salesforce, google, facebook

### top huggingface
- https://huggingface.co/runwayml/stable-diffusion-v1-5  10300
- https://huggingface.co/CompVis/stable-diffusion-v1-4    6210
- https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 4510
- https://huggingface.co/WarriorMama777/OrangeMixs    3590
- https://huggingface.co/prompthero/openjourney   3001
- https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt 1790
- https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0  1360
- https://huggingface.co/openai/clip-vit-large-patch14    932
- https://huggingface.co/timbrooks/instruct-pix2pix   786
- https://huggingface.co/stablediffusionapi/deliberate-v2   707
- https://huggingface.co/Salesforce/blip-image-captioning-large   706
- https://huggingface.co/nlpconnect/vit-gpt2-image-captioning 683
- https://huggingface.co/briaai/RMBG-1.4  601
- https://huggingface.co/stabilityai/stable-video-diffusion-img2vid   557
- https://huggingface.co/CrucibleAI/ControlNetMediaPipeFace   510
- https://huggingface.co/google/vit-base-patch16-224  453
- https://huggingface.co/stabilityai/stable-zero123   432
- https://huggingface.co/ali-vilab/text-to-video-ms-1.7b  431
- https://huggingface.co/cerspense/zeroscope_v2_576w  420
- https://huggingface.co/facebook/detr-resnet-50  418
- https://huggingface.co/Salesforce/blip-image-captioning-base    337
- https://huggingface.co/lambdalabs/sd-image-variations-diffusers 330
- https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K    239
- https://huggingface.co/acheong08/f222 185







## SYNTHETIC DATASETS
Synthetic data generation involves creating artificial images or videos that mimic real-world data. This technique is particularly useful in scenarios where collecting real-world data is challenging, expensive, or constrained by privacy and ethical considerations.

### How Synthetic Datasets Are Generated
- **3D Rendering**: Using 3D modeling software to create scenes and objects that are then rendered into images or videos. This method is commonly used for generating datasets for autonomous vehicle training, where diverse traffic scenarios can be simulated.
- **Generative Adversarial Networks (GANs**: GANs can generate highly realistic images by learning the distribution of a dataset. They are particularly useful for augmenting datasets with rare but critical cases, such as medical imaging datasets with rare diseases.
- **Data Augmentation Techniques**: While not generating completely new data, augmentation techniques such as flipping, rotation, scaling, and color variation can create variations of existing data, effectively increasing the size and diversity of a dataset.
- **Simulation Environments**: Simulated environments, often used in robotics and autonomous vehicles, can generate vast amounts of synthetic data. These environments simulate real-world physics and interactions, providing a rich source of diverse data.





# IMAGE RECOGINTION:
- ZASTOSOWANIE IMAGE TRANSFORMERS WYDAJE SIE BYC MEGA: [https://paperswithcode.com/sota/image-classification-on-cifar-10] & [https://paperswithcode.com/paper/an-image-is-worth-16x16-words-transformers-1]
- (Vision Transformer - Pytorch)[https://github.com/lucidrains/vit-pytorch]


# Stable diffusion UI
- https://github.com/comfyanonymous/ComfyUI
- https://civitai.com/ community


The most powerful and modular stable diffusion GUI and backend.

# TorchVision
Sprawdz!!!