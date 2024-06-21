#include "bvh.h"
#include <iostream>

/**
 * Given full mesh info (all triangles), create all AABBs
 * and build MTBVH on CPU.
 * This MTBVH will be copied to GPU explicitly in pathtrace.cu
 * 
 * @param vertices All triangle positions
 * @param boundingBoxes Output param for AABBs
 * @param BVHNodes Output param nodes corresponding to AABBs
 * 
 * @return BVHSize, number of nodes in the BVH tree, = (#triangles * 2 -1)
 */
int BVHBuilder::build(
    const std::vector<glm::vec3>& vertices,
    std::vector<AABB>& boundingBoxes,
    std::vector<std::vector<MTBVHNode>>& BVHNodes
) {
    std::cout << "[BVH building...]" << std::endl;
    int numPrims = int(vertices.size() / 3);
    int BVHSize = numPrims * 2 - 1;

    std::vector<PrimInfo> primInfo(numPrims);
    std::vector<NodeInfo> nodeInfo(BVHSize);
    boundingBoxes.resize(BVHSize);

    for (int i = 0; i < numPrims; i++) {
        primInfo[i].primId = i;
        primInfo[i].bound = AABB(vertices[i * 3 + 0], vertices[i * 3 + 1], vertices[i * 3 + 2]);
        primInfo[i].center = primInfo[i].bound.center();
    }

    // Use array stack just for faster
    std::vector<BuildInfo> stack(BVHSize);
    int stackTop = 0;
    stack[stackTop++] = { 0, 0, numPrims - 1 };  // { offset, start, end }

    const int NumBuckets = 16;
    // Use non-recursive top-down approach to build BVH data directly flattened
    int depth = 0;
    while (stackTop) {
        depth = std::max(depth, stackTop);
        stackTop--;
        // offset is the idx of current node being built in BVHNodes[]
        int offset = stack[stackTop].offset;
        int start = stack[stackTop].start;
        int end = stack[stackTop].end;

        int numSubPrims = end - start + 1;  // #tri taken care by current node
        int nodeSize = numSubPrims * 2 - 1;  // #nodes in the current-node rooted subtree
        bool isLeaf = nodeSize == 1;
        nodeInfo[offset] = { isLeaf, isLeaf ? primInfo[start].primId : nodeSize };

        AABB nodeBound, centerBound;
        /// traverse all tri owned by this node, combine their AABBs together
        /// NOTE: AABB operator() (const AABB& rhs) combines 2 AABBs
        for (int i = start; i <= end; i++) {
            nodeBound = nodeBound(primInfo[i].bound);
            centerBound = centerBound(primInfo[i].center);
        }
        boundingBoxes[offset] = nodeBound;

        //std::cout << std::setw(10) << offset << " " << start << " " << end << " " << nodeBound.toString() << "\n";

        if (isLeaf) {
            continue;
        }

        int splitAxis = centerBound.longestAxis();

        if (nodeSize == 3) {  // when we look at 2 triangles (when end-start=1)
            if (primInfo[start].center[splitAxis] > primInfo[end].center[splitAxis]) {
                std::swap(primInfo[start], primInfo[end]);
            }
            // push_back 2 leaf AABBs
            boundingBoxes[offset + 1] = primInfo[start].bound;
            boundingBoxes[offset + 2] = primInfo[end].bound;
            nodeInfo[offset + 1] = { true, primInfo[start].primId };
            nodeInfo[offset + 2] = { true, primInfo[end].primId };
        }

        /// otherwise, this node looks at many triangles
        /// and divides them into Buckets
        AABB bucketBounds[NumBuckets];
        int bucketCounts[NumBuckets];
        memset(bucketCounts, 0, sizeof(bucketCounts));

        float dimMin = centerBound.minPos[splitAxis];
        float dimMax = centerBound.maxPos[splitAxis];

        /// Evenly spaced buckets
        /// Suppose dimMin=0, dimMax=10, NumBuckets=5, and we are on x axis (splitAxis=0).
        /// Then 0:{0,2}, 1:{2,4}, ..., 4:{8,10}
        for (int i = start; i <= end; i++) {
            int bid = glm::clamp(
                int((primInfo[i].center[splitAxis] - dimMin) / (dimMax - dimMin) * NumBuckets),
                0, NumBuckets - 1);
            /// AABB of current bucket; #triangles in the bucket
            bucketBounds[bid] = bucketBounds[bid](primInfo[i].bound);
            bucketCounts[bid]++;
        }

        AABB lBounds[NumBuckets];
        AABB rBounds[NumBuckets];
        int countPrefix[NumBuckets];
        /// Continue the example setup above.
        /// 
        /// lBounds accumulate from the left:
        /// 0:{0,2}, 1:{0,4}, 2:{0,6}, 3:{0,8}, 4:{0,10}
        /// Similar happens for rBounds
        /// 0:{0,10}, 1:{2,10}, 2:{4,10}, 3:{6,10}, 4:{8,10}
        /// countPrefix stores the prefix sum of #triangles
        /// 0:|{0,2}|, 1:|{0,4}|, 2:|{0,6}|, 3:|{0,8}|, 4:|{0,10}|
        lBounds[0] = bucketBounds[0];
        rBounds[NumBuckets - 1] = bucketBounds[NumBuckets - 1];
        countPrefix[0] = bucketCounts[0];
        for (int i = 1, j = NumBuckets - 2; i < NumBuckets; i++, j--) {
            lBounds[i] = lBounds[i](bucketBounds[i - 1]);
            rBounds[j] = rBounds[j](bucketBounds[j + 1]);
            countPrefix[i] = countPrefix[i - 1] + bucketCounts[i];
        }

        float minSAH = FLT_MAX;
        int divBucket = 0;
        /// Use SAH heruistic to decide where to split the bucket
        /// divBucket : where to split
        /// 0 : {0,2}+{2,10}
        /// 3 : {0,8}+{8,10}
        /// SAH is the weighted by number of triangles
        for (int i = 0; i < NumBuckets - 1; i++) {
            float SAH = glm::mix(lBounds[i].surfaceArea(), rBounds[i + 1].surfaceArea(),
                float(countPrefix[i]) / numSubPrims);
            if (SAH < minSAH) {
                minSAH = SAH;
                divBucket = i;
            }
        }

        std::vector<PrimInfo> temp(numSubPrims);
        memcpy(temp.data(), primInfo.data() + start, numSubPrims * sizeof(PrimInfo));
        /// Continue the above example setup
        /// Suppose we divide at divBucket=3, which is {0,8}+{8,10}.
        /// Also, suppose 15 triangles should go to {0,8}, 5 will go to {8,10}.
        /// 
        /// But these 20 triangles in primInfo[start:start+20] is not sorted according to dim.x,
        /// and we don't want to sort.
        /// And this workaround can update primInfo[start:start+20] correctly.
        /// The 15 triangles for {0,8} reside in primInfo[start:start+15] and all have smaller dim.x;
        /// The 5 triangles for {8,10} reside in primInfo[start+15:start+20] and all have bigger dim.x;
        /// 
        /// NOTE that primInfo[start:start+15] isn't sorted by dim.x, and it makes no sense to do so.
        /// Next subdivision may work on dim.y
        int divPrim = start, divEnd = end;
        for (int i = 0; i < numSubPrims; i++) {
            int bid = glm::clamp(
                int((temp[i].center[splitAxis] - dimMin) / (dimMax - dimMin) * NumBuckets),
                0, NumBuckets - 1);
            if (bid <= divBucket) {
                primInfo[divPrim++] = temp[i];
            }
            else {
                primInfo[divEnd--] = temp[i];
            }
        }
        //divPrim = countPrefix[divBucket];
        divPrim = glm::clamp(divPrim - 1, start, end - 1);
        int lSize = 2 * (divPrim - start + 1) - 1;  // #nodes in my left subtree

        // Process left subtree first: depth = 54; right subtree first: depth = 61;
        //stack[stackTop++] = { offset + 1, start, divPrim };  // right processed first
        stack[stackTop++] = { offset + 1 + lSize, divPrim + 1, end };  // right subtree
        stack[stackTop++] = { offset + 1, start, divPrim };  // left processed first
    }

    std::cout << "\t[Size = " << BVHSize << ", depth = " << depth << "]" << std::endl;
    /// After creating all AABBs and knowing info of all N nodes,
    /// build MTBVH, essentially a forest with 6 N-node BVH.
    buildMTBVH(boundingBoxes, nodeInfo, BVHSize, BVHNodes);
    return BVHSize;
}


/**
 * Build the Multi-Threaded BVH given all AABBs and node info.
 * 
 * @param boundingBoxes AABBs (already created), const
 * @param nodeInfo      All node info, const
 * @param BVHSize       N = #nodes in a single/traditional BVH tree
 * @param BVHNodes      Output, to be populated, size (6, N)
 */
void BVHBuilder::buildMTBVH(
    const std::vector<AABB>& boundingBoxes,
    const std::vector<NodeInfo>& nodeInfo,
    int BVHSize,
    std::vector<std::vector<MTBVHNode>>& BVHNodes
) {
    BVHNodes.resize(NUM_FACES);
    std::vector<int> stack(BVHSize);

    for (int i = 0; i < NUM_FACES; i++) {
        auto& nodes = BVHNodes[i];
        nodes.resize(BVHSize);

        int stackTop = 0;
        stack[stackTop++] = 0;
        int nodeIdNew = 0;  // ++ when we create a new node of this N-node tree
        int nodeIdOrig;  // to index nodeInfo[]
        while (stackTop) {
            nodeIdOrig = stack[--stackTop];
            bool isLeaf = nodeInfo[nodeIdOrig].isLeaf;
            int nodeSize = isLeaf ? 1 : nodeInfo[nodeIdOrig].primIdOrSize;

            /// NOTE: if we have M triangles and N nodes (N = 2M - 1),
            /// the last M nodes in nodeInfo[N] are leaf nodes.
            /// 
            /// But for nodes[N], we want them in DFS pre-order,
            /// and nodes in nodeInfo (also AABBs in boundingBoxes) are in level-order.
            /// @see page 47-48 of https://cs.uwaterloo.ca/~thachisu/tdf2015.pdf
            /// 
            nodes[nodeIdNew] = {
                isLeaf ? nodeInfo[nodeIdOrig].primIdOrSize : NullPrimitive,
                nodeIdOrig,  // boundingBoxId, see above.
                nodeIdNew + nodeSize  // nextNodeIfMiss, see above
            };
            nodeIdNew++;

            if (isLeaf) {
                continue;
            }
            bool isLeftLeaf = nodeInfo[nodeIdOrig + 1].isLeaf;
            int leftSize = isLeftLeaf ? 1 : nodeInfo[nodeIdOrig + 1].primIdOrSize;

            /// point to the left-child and right-child indices
            /// in boundingBoxes[] and stack[]
            int left = nodeIdOrig + 1;
            int right = nodeIdOrig + 1 + leftSize;

            int dim = i / 2;  // 01:x, 23:y, 45:z
            bool lesser = i & 1;  // 135:lesser
            bool actually_lesser = 
                boundingBoxes[left].center()[dim] < boundingBoxes[right].center()[dim];
            if (actually_lesser ^ lesser) {
                std::swap(left, right);
            }

            /// process left first
            stack[stackTop++] = right;
            stack[stackTop++] = left;
        }
    }
}
