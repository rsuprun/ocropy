from setuptools import setup, find_packages


setup(
    name = 'ocropy_segmenter',
    version = '0.1.3',
    url = "https://github.com/rsuprun/ocropy",
    author = "Thomas Breuel (modified by Robin Suprun)",
    author_email = "robin.suprun@fraserhealth.ca",
    description = "Simpliefied text segmentation from Ocropy package",
    packages = find_packages(),
    install_requires=["pillow>=7.1",
                      "opencv-python>=4.0",
                      "numpy>=1.18",
                      "scipy>=1.4",
                      "matplotlib>=3.1",
                      "imageio>=2.8"],

    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Win32 (MS Windows)'
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Multimedia :: Graphics :: Capture',
    ],

)


