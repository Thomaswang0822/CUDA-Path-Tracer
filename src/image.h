#pragma once

#include <glm/glm.hpp>
#include <string>

using namespace std;

class Image {
private:
    glm::vec3 *pixels;

public:
    int xSize;
    int ySize;

    Image(int x, int y);
    Image(const std::string& filename);
    ~Image();
    void setPixel(int x, int y, const glm::vec3 &pixel);
    void savePNG(const std::string &baseFilename);
    void saveHDR(const std::string &baseFilename);

    /** Helpers */
    size_t byteSize() {
        return sizeof(glm::vec3) * xSize * ySize;
    }

    glm::vec3* data() const {
        return pixels;
    }
};
