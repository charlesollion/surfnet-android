package org.surfrider.surfnet.detection.models

import com.google.gson.annotations.SerializedName

data class TrackerPosition(
    @SerializedName("lat")
    var lat: Double,
    @SerializedName("lng")
    var lng: Double,
    @SerializedName("date")
    var date: String
)