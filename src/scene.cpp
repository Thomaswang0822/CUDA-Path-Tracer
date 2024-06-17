#include <iostream>
#include "scene.h"
//#include <cstring>
#include <string>
#include <map>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

map<string, Material::Type> MaterialTypeTokenMap = {
    { "Lambertian", Material::Type::Lambertian},
    { "MetallicWorkflow", Material::Type::MetallicWorkflow },
    { "Dielectric", Material::Type::Dielectric },
    { "Light", Material::Type::Light }
};

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }
    while (fp_in.good()) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (tokens[0] == "Material") {
                loadMaterial(tokens[1]);
                cout << " " << endl;
            }
            else if (tokens[0] == "Object") {
                loadGeom(tokens[1]);
                cout << " " << endl;
            }
            else if (tokens[0] == "Camera") {
                loadCamera();
                cout << " " << endl;
            }
        }
    }
}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    }
    else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (line == "Sphere") {
                cout << "Creating new sphere..." << endl;
                newGeom.type = GeomType::SPHERE;
            }
            else if (line == "Cube") {
                cout << "Creating new cube..." << endl;
                newGeom.type = GeomType::CUBE;
            }
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = stoi(tokens[1]);
            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
        }

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (tokens[0] == "Translate") {
                newGeom.translation = glm::vec3(stof(tokens[1]), stof(tokens[2]), stof(tokens[3]));
            }
            else if (tokens[0] == "Rotate") {
                newGeom.rotation = glm::vec3(stof(tokens[1]), stof(tokens[2]), stof(tokens[3]));
            }
            else if (tokens[0] == "Scale") {
                newGeom.scale = glm::vec3(stof(tokens[1]), stof(tokens[2]), stof(tokens[3]));
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
        return 1;
    }
}

int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState& state = this->state;
    Camera& camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 7; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "Resolution") == 0) {
            camera.resolution.x = stoi(tokens[1]);
            camera.resolution.y = stoi(tokens[2]);
        }
        else if (tokens[0] == "FovY") {
            fovy = stof(tokens[1]);
        }
        else if (tokens[0] == "LensRadius") {
            camera.lensRadius = stof(tokens[1]);
        }
        else if (tokens[0] == "FocalDist") {
            camera.focalDist = stof(tokens[1]);
        }
        else if (tokens[0] == "Iterations") {
            state.iterations = stoi(tokens[1]);
        }
        else if (tokens[0] == "Depth") {
            state.traceDepth = stoi(tokens[1]);
        }
        else if (tokens[0] == "File") {
            state.imageName = tokens[1];
        }
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (tokens[0] == "Eye") {
            camera.position = glm::vec3(stof(tokens[1]), stof(tokens[2]), stof(tokens[3]));
        }
        else if (tokens[0] == "LookAt") {
            camera.lookAt = glm::vec3(stof(tokens[1]), stof(tokens[2]), stof(tokens[3]));
        }
        else if (tokens[0] == "Up") {
            camera.up = glm::vec3(stof(tokens[1]), stof(tokens[2]), stof(tokens[3]));
        }

        utilityCore::safeGetline(fp_in, line);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    fill(state.image.begin(), state.image.end(), glm::vec3());

    cout << "Loaded camera!" << endl;
    return 1;
}

int Scene::loadMaterial(string matId) {
    int id = atoi(matId.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    }
    else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 6; i++) {
            string line;
            utilityCore::safeGetline(fp_in, line);
            auto tokens = utilityCore::tokenizeString(line);
            if (tokens[0] == "Type") {
                newMaterial.type = MaterialTypeTokenMap[tokens[1]];
            }
            else if (tokens[0] == "BaseColor") {
                glm::vec3 baseColor(stof(tokens[1]), stof(tokens[2]), stof(tokens[3]));
                newMaterial.baseColor = baseColor;
            }
            else if (tokens[0] == "Metallic") {
                newMaterial.metallic = stof(tokens[1]);
            }
            else if (tokens[0] == "Roughness") {
                newMaterial.roughness = stof(tokens[1]);
            }
            else if (tokens[0] == "Ior") {
                newMaterial.ior = stof(tokens[1]);
            }
            else if (tokens[0] == "Emittance") {
                newMaterial.emittance = stof(tokens[1]);
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}
