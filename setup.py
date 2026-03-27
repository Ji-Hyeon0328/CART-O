from setuptools import setup, find_packages

# 패키지를 시스템에 등록하기 위한 설정 파일입니다.
setup(
    name="isaaclab_carto",
    version="1.0.0",
    author="shoko",
    description="Isaac Lab extension for Spot robot navigation research.",
    packages=find_packages(),
    author_email="shoko@ksim-mg-1", # 사용자 환경 기반
    install_requires=[
        "torch",
        "numpy",
    ],
    python_requires=">=3.10",
)