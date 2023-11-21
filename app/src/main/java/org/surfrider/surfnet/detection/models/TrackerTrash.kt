package org.surfrider.surfnet.detection.models

import com.google.gson.annotations.SerializedName

data class TrackerTrash(
    @SerializedName("date") var date: String? = null,
    @SerializedName("lat") var lat: Double? = null,
    @SerializedName("lng") var lng: Double? = null,
    @SerializedName("name") var name: String? = null
)