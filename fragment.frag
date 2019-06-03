#version 330 core

in vec3 pos;

out vec4 FragColor;

struct Point {
	float x;
	float y;
	float value;
};

#define ARRAY_SIZE 16
#define SCR_WIDTH 512
#define SCR_HEIGHT 512
#define STEP 128
uniform vec3 color;
uniform Point points[ARRAY_SIZE];

float bilinear(float x, float y, Point p1, Point p2, Point p3, Point p4) {

	float x1 = p1.x;
	float y1 = p1.y;
	float x2 = p2.x;
	float y2 = p3.y;
	
	float q11 = p1.value;
	float q21 = p2.value;
	float q12 = p3.value;
	float q22 = p4.value;

	return (q11 * (x2 - x) * (y2 - y) +
	        q21 * (x - x1) * (y2 - y) +
	        q12 * (x2 - x) * (y - y1) +
	        q22 * (x - x1) * (y - y1)
	       ) / ((x2 - x1) * (y2 - y1) + 0.0);
}

void main()
{

	int column = int(pos.x / STEP);
 	int row = int(pos.y / STEP);
	int column_count = int(SCR_WIDTH / STEP);
	int point1_index = row*column_count + column;
	int point2_index = point1_index + 1;
	int point3_index = (row + 1)*column_count + column;
	int point4_index = (row + 1)*column_count + column + 1;
	float result = bilinear(pos.x, pos.y, points[point1_index], points[point2_index], points[point3_index], points[point4_index]);

	// Point p = points[point4_index];
	// float result = p.value;

	// float result = pos.x * 1000;

	// FragColor = vec4(color, 1.0);
    FragColor = vec4(vec3(result), 1.0);
    // FragColor = vec4(pos, 1.0);
}