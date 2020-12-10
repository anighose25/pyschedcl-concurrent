typedef struct Vector
{
    __global uchar *ptr;
    int offset_first_element_in_bytes;
    int stride_x;
} Vector;


typedef struct Image
{
    __global uchar *ptr;
    int offset_first_element_in_bytes;
    int stride_x;
    int stride_y;
} Image;


typedef struct Tensor3D
{
    __global uchar *ptr;
    int offset_first_element_in_bytes;
    int stride_x;
    int stride_y;
    int stride_z;
} Tensor3D;


typedef struct Tensor4D
{
    __global uchar *ptr;
    int offset_first_element_in_bytes;
    int stride_x;
    int stride_y;
    int stride_z;
    int stride_w;
} Tensor4D;

typedef float ACC_DATA_TYPE;
typedef float4 ACC_DATA_TYPE4;
typedef float8 ACC_DATA_TYPE8;

inline Image update_image_from_tensor3D_workitem_ptr(__global uchar *ptr, uint offset_first_element_in_bytes, uint stride_x, uint step_x, uint stride_y, uint step_y, uint stride_z, uint step_z)
{
    Image img =
    {
        .ptr = ptr,
        .offset_first_element_in_bytes = offset_first_element_in_bytes,
        .stride_x = stride_x,
        .stride_y = stride_y
    };
    img.ptr += img.offset_first_element_in_bytes + get_global_id(0) * step_x + get_global_id(1) * step_y + get_global_id(2) * step_z;
    return img;
}
inline Tensor3D update_tensor3D_workitem_ptr(__global uchar *ptr, uint offset_first_element_in_bytes, uint stride_x, uint step_x, uint stride_y, uint step_y, uint stride_z, uint step_z)
{
    Tensor3D tensor =
    {
        .ptr = ptr,
        .offset_first_element_in_bytes = offset_first_element_in_bytes,
        .stride_x = stride_x,
        .stride_y = stride_y,
        .stride_z = stride_z
    };
    tensor.ptr += tensor.offset_first_element_in_bytes + get_global_id(0) * step_x + get_global_id(1) * step_y + get_global_id(2) * step_z;
    return tensor;
}
inline Tensor3D tensor3D_ptr_no_update(__global uchar *ptr, uint offset_first_element_in_bytes, uint stride_x, uint step_x, uint stride_y, uint step_y, uint stride_z, uint step_z)
{
    Tensor3D tensor =
    {
        .ptr = ptr,
        .offset_first_element_in_bytes = offset_first_element_in_bytes,
        .stride_x = stride_x,
        .stride_y = stride_y,
        .stride_z = stride_z
    };
    return tensor;
}


inline __global const uchar *vector_offset(const Vector *vec, int x)
{
    return vec->ptr + x * vec->stride_x;
}







inline __global uchar *offset(const Image *img, int x, int y)
{
    return img->ptr + x * img->stride_x + y * img->stride_y;
}
inline __global const uchar *tensor3D_offset(const Tensor3D *tensor, int x, int y, int z)
{
    return tensor->ptr + x * tensor->stride_x + y * tensor->stride_y + z * tensor->stride_z;
}
inline __global const uchar *tensor4D_offset(const Tensor4D *tensor, int x, int y, int z, int w)
{
    return tensor->ptr + x * tensor->stride_x + y * tensor->stride_y + z * tensor->stride_z + w * tensor->stride_w;
}
inline __global const uchar *tensor3D_index2ptr(const Tensor3D *tensor, uint width, uint height, uint depth, uint index)
{
    uint num_elements = width * height;

    const uint z = index / num_elements;

    index %= num_elements;

    const uint y = index / width;

    index %= width;

    const uint x = index;

    return tensor->ptr + x * tensor->stride_x + y * tensor->stride_y + z * tensor->stride_z + tensor->offset_first_element_in_bytes;
}


__kernel void pooling_layer_optimized_3(
    __global uchar *input_ptr, uint input_stride_x, uint input_step_x, uint input_stride_y, uint input_step_y, uint input_stride_z, uint input_step_z, uint input_offset_first_element_in_bytes,
    __global uchar *output_ptr, uint output_stride_x, uint output_step_x, uint output_stride_y, uint output_step_y, uint output_stride_z, uint output_step_z, uint output_offset_first_element_in_bytes)
{

    Tensor3D input = update_tensor3D_workitem_ptr(input_ptr, input_offset_first_element_in_bytes, input_stride_x, input_step_x, input_stride_y, input_step_y, input_stride_z, input_step_z);
    Tensor3D output = update_tensor3D_workitem_ptr(output_ptr, output_offset_first_element_in_bytes, output_stride_x, output_step_x, output_stride_y, output_step_y, output_stride_z, output_step_z);

    ACC_DATA_TYPE4 res;


    ({ ACC_DATA_TYPE8 data00 = vload8(0, (__global float *)tensor3D_offset(&input, 0, 0, 0));
        ACC_DATA_TYPE data01 = (ACC_DATA_TYPE)(*((__global float *)tensor3D_offset(&input, 0, 0, 0) + 8));
        ACC_DATA_TYPE8 data10 = vload8(0, (__global float *)tensor3D_offset(&input, 0, 1, 0));
        ACC_DATA_TYPE data11 = (ACC_DATA_TYPE)(*((__global float *)tensor3D_offset(&input, 0, 1, 0) + 8));
        ACC_DATA_TYPE8 data20 = vload8(0, (__global float *)tensor3D_offset(&input, 0, 2, 0));
        ACC_DATA_TYPE data21 = (ACC_DATA_TYPE)(*((__global float *)tensor3D_offset(&input, 0, 2, 0) + 8));
        \
        data00 = (data00);
        data01 = (data01); 
        data10 = (data10); 
        data11 = (data11); 
        data20 = (data20); 
        data21 = (data21); 
        \
        ACC_DATA_TYPE8 values00 = (ACC_DATA_TYPE8)(data00.s01223445); 
        ACC_DATA_TYPE4 values01 = (ACC_DATA_TYPE4)(data00.s667, data01); 
        ACC_DATA_TYPE8 values10 = (ACC_DATA_TYPE8)(data10.s01223445); 
        ACC_DATA_TYPE4 values11 = (ACC_DATA_TYPE4)(data10.s667, data11); 
        ACC_DATA_TYPE8 values20 = (ACC_DATA_TYPE8)(data20.s01223445); 
        ACC_DATA_TYPE4 values21 = (ACC_DATA_TYPE4)(data20.s667, data21); 
        \
        values00 = (fmax((values00), (values10))); 
        values01 = (fmax((values01), (values11))); 
        values00 = (fmax((values00), (values20))); 
        values01 = (fmax((values01), (values21))); 
        res = (fmax(((ACC_DATA_TYPE4)(values00.s036, values01.s1)), ((ACC_DATA_TYPE4)(values00.s147, values01.s2)))); 
        res = (fmax((res), ((ACC_DATA_TYPE4)(values00.s25, values01.s03)))); 
    });
    vstore4((convert_float4((res))), 0, (__global float *)output.ptr);
}

