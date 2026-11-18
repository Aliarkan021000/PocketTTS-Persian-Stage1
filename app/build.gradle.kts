plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "ir.pockettts.farsi"
    compileSdk = 35

    defaultConfig {
        applicationId = "ir.pockettts.farsi"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0.0-stage1"

        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    buildFeatures {
        compose = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.14"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {
    // ONNX Runtime Android (برای اجرای g2p_encoder/decoder، flow_lm_step و mimi_decoder_step_kv)
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.2")

    // Jetpack Compose & Material 3
    implementation(platform("androidx.compose:compose-bom:2024.06.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.activity:activity-compose:1.9.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.4")
    implementation("androidx.core:core-ktx:1.13.1")

    // Coroutines for background synthesis
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
}
