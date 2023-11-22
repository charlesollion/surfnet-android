package org.surfrider.surfnet.detection.env

import android.graphics.PointF
import android.graphics.RectF
import kotlin.math.pow
import kotlin.math.sqrt

object MathUtils {

    @JvmStatic
    fun solveLinearSumAssignment(costMatrix: Array<DoubleArray>): List<Pair<Int, Int>> {
        val numRows = costMatrix.size
        val numCols = costMatrix[0].size

        val maxDim = maxOf(numRows, numCols)

        // Step 1: Subtract the minimum value from each row
        for (i in 0 until numRows) {
            val minVal = costMatrix[i].minOrNull() ?: 0.0
            if(minVal == Double.MAX_VALUE) break
            for (j in 0 until numCols) {
                costMatrix[i][j] -= minVal
            }
        }

        // Step 2: Subtract the minimum value from each column
        for (j in 0 until numCols) {
            val minVal = (0 until numRows).map { costMatrix[it][j] }.minOrNull() ?: 0.0
            if(minVal == Double.MAX_VALUE) break
            for (i in 0 until numRows) {
                costMatrix[i][j] -= minVal
            }
        }

        val assignments = mutableListOf<Pair<Int, Int>>()

        // Step 3: Find the optimal assignments
        val coveredRows = BooleanArray(numRows)
        val coveredCols = BooleanArray(numCols)
        while (true) {
            var minUncoveredValue = Double.MAX_VALUE

            // Find the minimum uncovered value
            for (i in 0 until numRows) {
                for (j in 0 until numCols) {
                    if (!coveredRows[i] && !coveredCols[j] && costMatrix[i][j] < minUncoveredValue) {
                        minUncoveredValue = costMatrix[i][j]
                    }
                }
            }

            if (minUncoveredValue == Double.MAX_VALUE) {
                break
            }

            // Cover rows and uncover columns with minimum value
            for (i in 0 until numRows) {
                for (j in 0 until numCols) {
                    if (!coveredRows[i] && !coveredCols[j] && costMatrix[i][j] == minUncoveredValue) {
                        assignments.add(Pair(i, j))
                        coveredRows[i] = true
                        coveredCols[j] = true
                    }
                }
            }
        }

        return assignments
    }

    @JvmStatic
    fun calculateIoU(rect1: RectF, rect2: RectF): Float {
        // Calculate the intersection of the two rectangles
        val intersection = RectF()
        intersection.setIntersect(rect1, rect2)

        // Calculate the area of intersection
        val intersectionArea = if (intersection.isEmpty) 0.0f else intersection.width() * intersection.height()

        // Calculate the area of the individual rectangles
        val areaRect1 = rect1.width() * rect1.height()
        val areaRect2 = rect2.width() * rect2.height()

        // Calculate IoU
        return intersectionArea / (areaRect1 + areaRect2 - intersectionArea)
    }

    fun malDist(p1: PointF, p2: PointF, dx: PointF): Float {
        val diffX = p1.x - p2.x
        val diffY = p1.y - p2.y
        val mahalanobisDistanceX = (diffX / (1.0F + dx.x)).pow(2)
        val mahalanobisDistanceY = (diffY / (1.0F + dx.y)).pow(2)

        // Calculate the Mahalanobis distance by summing the squared components
        return sqrt(mahalanobisDistanceX + mahalanobisDistanceY)
    }

}