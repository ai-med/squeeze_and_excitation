import setuptools

setuptools.setup(name="squeeze-and-excitation",
                 version="1.0",
                 url="https://github.com/abhi4ssj/squeeze_and_excitation",
                 author="Shayan Ahmad Siddiqui and Abhijit Roy Guha",
                 author_email="shayan.siddiqui89@gmail.com",
                 description="Squeeze and Excitation pytorch implementation",
                 packages=setuptools.find_packages(),
                 install_requires=['numpy>=1.14.0', 'torch>=1.0.0'],
                 python_requires='>=3.5')
