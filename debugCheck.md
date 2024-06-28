# DEBUG Log


## pathtrace.cu

- [X] computeIntersection
- [X] genRayFromCamera
- [X] shadeSegment
- [X] finalGather

## main

- [X] everything

## bvh
- [X] everything

## intersection

- [X] everything

## mathUtil

- [X] everything

## utilityCore
- [X] everything

## scene.h and scene.cu

- [X] total swap check: TOTAL BLACK

### DevScene

- [X] change field

```cpp
    createDevData(const Scene& scene);  // BUG TYPO
    freeDevData();


    getMTBVHId;
    getPrimitiveNormal;
    getIntersecGeomInfo;
    intersectPrimitive;
    intersectPrimitive;
    intersectPrimitiveDetailed;
    intersect;
    testOcclusion(glm::vec3 x, glm::vec3 y);  // BUG dist - 1e-4
    void visualizedIntersect;
    sampleDirectLight;  // BUG float power = Math::luminance(radiance) /*/ (area * TWO_PI)*/;
```

DONE

### Scene

- [X] change field

- [X] private member:

```cpp
    createLightSampler;
    loadMaterial;  // BUG: still need emittance; Material is wrong!
    loadModel;
    loadCamera;
```

- [ ] public member:

```cpp
    Scene(const std::string& filename);
    ~Scene;  // clear() or noop?
    buildDevData;  // BUG: same emittance bug, see below
    clear;
```

```
        
        glm::vec3 radianceUnitArea = material.baseColor * material.emittance;  // emittance dropped
        float powerUnitArea = Math::luminance(radianceUnitArea);  // no 2PI
```
## sampler

- [X] RNG
- [X] LightSampler

- Something is wrong with `sampleDirectLight()`, but all else are correct.

```cpp
    float power = Math::luminance(radiance) /*/ (area * TWO_PI)*/;
```

## camera

- [X] definition
- [X] loadCamera
- [X] camera instance in pathtrace.cu
    - [X] generateRayFromCamera()
- [X] camera control in main.cpp

## materials

Something wrong with dielectric

- BSDF is correct (vec3 0)
- pdf is wrong (should be 0 instead of 1)
- sample is the same

Have to be helper functions:

- Schilick approximation is wrong!
- But even after "correcting" it, teapot still look weird, not as good as precise Fresnel.