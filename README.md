
[![Issues][issues-shield]][issues-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="">
    <img src="https://avatars0.githubusercontent.com/u/36377110?s=200&v=4" width=80 height=80>
  </a>
  <h3 align="center">EduGAN</h3>

  <p align="center">
    Education Data GAN
    <br />
    <a href="https://github.com/pbaut002/EduGAN"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/pbaut002/EduGAN">View Demo</a>
    ·
    <a href="https://github.com/pbaut002/EduGAN/issues">Report Bug</a>
    ·
    <a href="https://github.com/pbaut002/EduGAN/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)



<!-- ABOUT THE PROJECT -->
## About The Project

Generative Adversarial Network designed for education data generation.

Create synthetic classes that resemble a real class without recreating actual samples.

Distribute class datasets without compromising privacy and security of actual data.



### Built With

* [Python]()
* [TensorFlow]()
* [Pandas]()
* [numpy]() 
* [Others can be found in requirements.txt]() 


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Installation

1. Clone the repo
```sh
git clone https://github.com/ILXL/EduGAN
```
2. Install NPM packages
```sh
pip3 install -r requirements.txt
```



<!-- USAGE EXAMPLES -->
## Usage

<ol> 
  <li>Convert raw data into CSV format</li>
  <li>Clean CSV data and keywords</li>
    * Example available in DataProcessor.py
  <li>Add JSON parameters into DataInformation.json</li>
  <li>Change keyword in GAN_Test.py to desired key from DataInformation.json</li>
  <li>Run GAN_Test.py</li>
    
</ol>



<!-- ROADMAP -->
## Roadmap

<ul>
  <li> Add support for categorical features</li>
  <li> Create UI for visualization and adjustment in-between training
</ul>


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Peter Bautista - peter.g.bautista@gmail.com

Project Link: [https://github.com/pbaut002/EduGAN](https://github.com/pbaut002/EduGAN)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* [Paul Salvador Inventado, PhD]()






<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[issues-shield]: https://img.shields.io/github/issues/pbaut002/EduGAN.svg?style=flat-square
[issues-url]: https://github.com/pbaut002/EduGAN/issues


