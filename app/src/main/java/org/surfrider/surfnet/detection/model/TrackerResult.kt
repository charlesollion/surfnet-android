package org.surfrider.surfnet.detection.model

import com.google.gson.annotations.SerializedName
import java.io.File


data class TrackerResult(
//    @SerializedName("id") var id: String? = null,
//    @SerializedName("date") var date: String? = null,
    @SerializedName("move") var move: String? = null,
    @SerializedName("bank") var bank: String? = null,
    @SerializedName("trackingMode") var trackingMode: String? = null,
    @SerializedName("files") var files: ArrayList<File> = arrayListOf(),
    @SerializedName("trashes") var trashes: ArrayList<TrackerTrash> = arrayListOf(),
    @SerializedName("positions") var positions: ArrayList<TrackerPosition> = arrayListOf(),
    @SerializedName("comment") var comment: String? = null
)
