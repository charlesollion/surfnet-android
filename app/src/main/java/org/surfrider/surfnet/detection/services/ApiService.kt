package org.surfrider.surfnet.detection.services

import retrofit2.Response
import retrofit2.http.*
import java.io.File

interface ApiService {

    @POST("login")
    suspend fun login(): Response<Any>

    @Multipart
    @POST("trace")
    suspend fun sendTrace(
        traceId: String,
        @Part file: File
    ): Response<Any>
}