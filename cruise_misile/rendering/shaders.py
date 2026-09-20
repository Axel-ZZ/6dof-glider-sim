VERT_SHADER = """
#version 330 core

in vec3 in_position;
in vec3 in_normal;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat3 u_normal_mat;

out vec3 v_normal;
out vec3 v_frag_pos;

void main() {
    vec4 world_pos  = u_model * vec4(in_position, 1.0);
    v_frag_pos      = world_pos.xyz;
    v_normal        = normalize(u_normal_mat * in_normal);
    gl_Position     = u_proj * u_view * world_pos;
}
"""

FRAG_SHADER = """
#version 330 core

in vec3 v_normal;
in vec3 v_frag_pos;

uniform vec3 u_light_pos;
uniform vec3 u_view_pos;
uniform vec3 u_color;

out vec4 frag_color;

void main() {
    // Ambient
    float ambient_strength = 0.48;
    vec3 ambient = ambient_strength * vec3(1.0);

    // Diffuse
    vec3 light_dir = normalize(u_light_pos - v_frag_pos);
    float diff     = max(dot(v_normal, light_dir), 0.0);
    vec3 diffuse   = diff * vec3(1.0);

    // Specular (Blinn-Phong)
    float spec_strength = 0.45;
    vec3 view_dir  = normalize(u_view_pos - v_frag_pos);
    vec3 half_dir  = normalize(light_dir + view_dir);
    float spec     = pow(max(dot(v_normal, half_dir), 0.0), 64.0);
    vec3 specular  = spec_strength * spec * vec3(1.0);

    vec3 result = (ambient + diffuse + specular) * u_color;
    frag_color  = vec4(result, 1.0);
}
"""

GRID_VERT_SHADER = """
#version 330 core

in vec3 in_position;

uniform mat4 u_view;
uniform mat4 u_proj;

out vec3 v_near;
out vec3 v_far;

vec3 unprojectPoint(float x, float y, float z, mat4 view, mat4 proj) {
    mat4 viewProjInv = inverse(proj * view);
    vec4 unprojectedPoint =  viewProjInv * vec4(x, y, z, 1.0);
    return unprojectedPoint.xyz / unprojectedPoint.w;
}

void main() {
    v_near = unprojectPoint(in_position.x, in_position.y, -1.0, u_view, u_proj);
    v_far  = unprojectPoint(in_position.x, in_position.y, 1.0, u_view, u_proj);
    gl_Position = vec4(in_position, 1.0);
}
"""

GRID_FRAG_SHADER = """
#version 330 core

in vec3 v_near;
in vec3 v_far;

uniform mat4 u_view;
uniform mat4 u_proj;

out vec4 frag_color;

vec4 grid(vec3 fragPos3D, float scale) {
    vec2 coord = fragPos3D.xy * scale; // X, Y grid for Z-up
    vec2 derivative = fwidth(coord);
    vec2 grid = abs(fract(coord - 0.5) - 0.5) / derivative;
    float line = min(grid.x, grid.y);
    float color = 1.0 - min(line, 1.0);
    vec4 res = vec4(vec3(0.2), color); // Dark gray lines
    if(fragPos3D.x > -0.1 * scale && fragPos3D.x < 0.1 * scale) res.z = 1.0;
    if(fragPos3D.y > -0.1 * scale && fragPos3D.y < 0.1 * scale) res.x = 1.0;
    return res;
}

float computeDepth(vec3 pos) {
    vec4 clip_space_pos = u_proj * u_view * vec4(pos.xyz, 1.0);
    return (clip_space_pos.z / clip_space_pos.w);
}

void main() {
    float t = -v_near.z / (v_far.z - v_near.z); // Intersect with Z=0
    if (t < 0.0) discard;

    vec3 fragPos3D = v_near + t * (v_far - v_near);
    gl_FragDepth = (computeDepth(fragPos3D) + 1.0) / 2.0;

    // Use actual distance from camera to find fading
    float dist = length(v_near - fragPos3D);
    float fading = exp(-dist * 0.005); // Fade out over ~200-500 meters

    frag_color = (grid(fragPos3D, 1.0) + grid(fragPos3D, 0.1));
    frag_color.a *= fading;
}
"""

LINE_VERT_SHADER = """
#version 330 core

in vec3 in_position;
in vec3 in_color;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;

out vec3 v_color;

void main() {
    v_color = in_color;
    gl_Position = u_proj * u_view * u_model * vec4(in_position, 1.0);
}
"""

LINE_FRAG_SHADER = """
#version 330 core

in vec3 v_color;
out vec4 frag_color;

void main() {
    frag_color = vec4(v_color, 1.0);
}
"""
UI_VERT_SHADER = """
#version 330 core

in vec2 in_position;
in vec2 in_texcoord;

out vec2 v_texcoord;

void main() {
    v_texcoord = in_texcoord;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

UI_FRAG_SHADER = """
#version 330 core

in vec2 v_texcoord;
uniform sampler2D u_texture;

out vec4 frag_color;

void main() {
    frag_color = texture(u_texture, v_texcoord);
}
"""
