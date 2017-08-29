# opencv-android-sdk

opencv-3.2.0-android-sdk，android studio版本。<br/>
本项目旨在帮助用户，在android上快速使用opencv，做一些算法测试。

<h2>解决问题</h2>
opencv官方提供的android sdk，还是eclipse版本，而且默认需要安装opencv manager。<br/>

本项目，将官方最新版本转为studio版本，并且不需要安装opencv manager。<br/>

只需要简单配置，就可以在android项目中方便的使用opencv函数。<br/>

另外项目中，自带了一个官网的人脸检测例子，功参考。


<h2>使用方法</h2>

Gradle dependency

    repositories {
        jcenter()
        maven { url "https://jitpack.io" }
    }

    dependencies {
        compile 'com.github.hailindai:opencv-android-sdk:v1.0.0'
    }
	
	//加载opencv库。
    OpenCVLoader.initDebug();




