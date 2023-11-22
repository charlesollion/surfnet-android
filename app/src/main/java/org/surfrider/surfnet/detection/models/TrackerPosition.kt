package org.surfrider.surfnet.detection.models

import com.google.gson.annotations.SerializedName

data class TrackerPosition(
    @SerializedName("lat")
    var lat: Double? = null,
    @SerializedName("lng")
    var lng: Double? = null,
    @SerializedName("date")
    var date: String? = null
)