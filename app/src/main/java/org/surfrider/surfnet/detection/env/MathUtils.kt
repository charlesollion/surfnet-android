package org.surfrider.surfnet.detection.env

object MathUtils {

    @JvmStatic
    fun solveLinearSumAssignment(costMatrix: Array<DoubleArray>): List<Pair<Int, Int>> {
        val numRows = costMatrix.size
        val numCols = costMatrix[0].size

        val maxDim = maxOf(numRows, numCols)

        // Step 1: Subtract the minimum value from each row
        for (i in 0 until numRows) {
            val minVal = costMatrix[i].minOrNull() ?: 0.0
            for (j in 0 until numCols) {
                costMatrix[i][j] -= minVal
            }
        }

        // Step 2: Subtract the minimum value from each column
        for (j in 0 until numCols) {
            val minVal = (0 until numRows).map { costMatrix[it][j] }.minOrNull() ?: 0.0
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
}