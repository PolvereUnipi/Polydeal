// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) XXXX - YYYY by the polyDEAL authors
//
// This file is part of the polyDEAL library.
//
// Detailed license information governing the source code
// can be found in LICENSE.md at the top level directory.
//
// -----------------------------------------------------------------------------

#include <deal.II/base/function.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <agglomeration_handler.h>
#include <fe_agglodgp.h>
#include <poly_utils.h>

#include <algorithm>
#include <chrono>
#include <fstream>

#define HEX TRUE

enum class GridType
{
  grid_generator, // hyper_cube or hyper_ball
  unstructured    // square generated with gmsh, unstructured
};



enum class PartitionerType
{
  metis,
  rtree,
  no_partition
};



enum SolutionType
{
  linear,      // x+y-1
  quadratic,   // x^2+y^2-1
  product,     // xy(x-1)(y-1)
  product_sine // sin(pi*x)*sin(pi*y)
};



template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide(const SolutionType &sol_type = SolutionType::linear)
    : Function<dim>()
  {
    solution_type = sol_type;
  }

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double>           &values,
             const unsigned int /*component*/) const override;

private:
  SolutionType solution_type;
};



template <int dim>
void
RightHandSide<dim>::value_list(const std::vector<Point<dim>> &points,
                               std::vector<double>           &values,
                               const unsigned int /*component*/) const
{
  if (solution_type == SolutionType::linear)
    {
      for (unsigned int i = 0; i < values.size(); ++i)
        values[i] = 0.; // Laplacian of linear function
    }
  else if (solution_type == SolutionType::quadratic)
    {
      for (unsigned int i = 0; i < values.size(); ++i)
        values[i] = -4.; // quadratic (radial) solution
    }
  else if (solution_type == SolutionType::product)
    {
      for (unsigned int i = 0; i < values.size(); ++i)
        values[i] = -2. * points[i][0] * (points[i][0] - 1.) -
                    2. * points[i][1] * (points[i][1] - 1.);
    }
  else if (solution_type == SolutionType::product_sine)
    {
      // 2pi^2*sin(pi*x)*sin(pi*y)
      for (unsigned int i = 0; i < values.size(); ++i)
        values[i] = 2. * numbers::PI * numbers::PI *
                    std::sin(numbers::PI * points[i][0]) *
                    std::sin(numbers::PI * points[i][1]);
    }
  else
    {
      Assert(false, ExcNotImplemented());
    }
}



template <int dim>
class SolutionLinear : public Function<dim>
{
public:
  SolutionLinear()
    : Function<dim>()
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double>           &values,
             const unsigned int /*component*/) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim>  &p,
           const unsigned int component = 0) const override;
};

template <int dim>
double
SolutionLinear<dim>::value(const Point<dim> &p, const unsigned int) const
{
  double sum = 0;
  for (unsigned int d = 0; d < dim; ++d)
    sum += p[d];

  return sum - 1; // p[0]+p[1]+p[2]-1
}

template <int dim>
Tensor<1, dim>
SolutionLinear<dim>::gradient(const Point<dim> &p, const unsigned int) const
{
  (void)p;
  Tensor<1, dim> return_value;
  for (unsigned int d = 0; d < dim; ++d)
    return_value[d] = 0.;
  return return_value;
}


template <int dim>
void
SolutionLinear<dim>::value_list(const std::vector<Point<dim>> &points,
                                std::vector<double>           &values,
                                const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = this->value(points[i]);
}



template <int dim>
class SolutionQuadratic : public Function<dim>
{
public:
  SolutionQuadratic()
    : Function<dim>()
  {
    Assert(dim == 2, ExcNotImplemented());
  }

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double>           &values,
             const unsigned int /*component*/) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim>  &p,
           const unsigned int component = 0) const override;
};

template <int dim>
double
SolutionQuadratic<dim>::value(const Point<dim> &p, const unsigned int) const
{
  return p[0] * p[0] + p[1] * p[1] - 1; // ball, radial solution
}

template <int dim>
Tensor<1, dim>
SolutionQuadratic<dim>::gradient(const Point<dim> &p, const unsigned int) const
{
  Tensor<1, dim> return_value;
  return_value[0] = 2. * p[0];
  return_value[1] = 2. * p[1];
  return return_value;
}


template <int dim>
void
SolutionQuadratic<dim>::value_list(const std::vector<Point<dim>> &points,
                                   std::vector<double>           &values,
                                   const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = this->value(points[i]);
}



template <int dim>
class SolutionProduct : public Function<dim>
{
public:
  SolutionProduct()
    : Function<dim>()
  {
    Assert(dim == 2, ExcNotImplemented());
  }

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double>           &values,
             const unsigned int /*component*/) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim>  &p,
           const unsigned int component = 0) const override;

  virtual void
  gradient_list(const std::vector<Point<dim>> &points,
                std::vector<Tensor<1, dim>>   &gradients,
                const unsigned int /*component*/) const override;
};

template <int dim>
double
SolutionProduct<dim>::value(const Point<dim> &p, const unsigned int) const
{
  return p[0] * (p[0] - 1.) * p[1] * (p[1] - 1.); // square
}

template <int dim>
Tensor<1, dim>
SolutionProduct<dim>::gradient(const Point<dim> &p, const unsigned int) const
{
  Tensor<1, dim> return_value;
  return_value[0] = (-1 + 2 * p[0]) * (-1 + p[1]) * p[1];
  return_value[1] = (-1 + 2 * p[1]) * (-1 + p[0]) * p[0];
  return return_value;
}


template <int dim>
void
SolutionProduct<dim>::value_list(const std::vector<Point<dim>> &points,
                                 std::vector<double>           &values,
                                 const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = this->value(points[i]);
}



template <int dim>
void
SolutionProduct<dim>::gradient_list(const std::vector<Point<dim>> &points,
                                    std::vector<Tensor<1, dim>>   &gradients,
                                    const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < gradients.size(); ++i)
    gradients[i] = this->gradient(points[i]);
}



template <int dim>
class SolutionProductSine : public Function<dim>
{
public:
  SolutionProductSine()
    : Function<dim>()
  {
    Assert(dim == 2, ExcNotImplemented());
  }

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double>           &values,
             const unsigned int /*component*/) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim>  &p,
           const unsigned int component = 0) const override;
};

template <int dim>
double
SolutionProductSine<dim>::value(const Point<dim> &p, const unsigned int) const
{
  return std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]);
}

template <int dim>
Tensor<1, dim>
SolutionProductSine<dim>::gradient(const Point<dim> &p,
                                   const unsigned int) const
{
  Tensor<1, dim> return_value;
  return_value[0] =
    numbers::PI * std::cos(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]);
  return_value[1] =
    numbers::PI * std::cos(numbers::PI * p[1]) * std::sin(numbers::PI * p[0]);
  return return_value;
}


template <int dim>
void
SolutionProductSine<dim>::value_list(const std::vector<Point<dim>> &points,
                                     std::vector<double>           &values,
                                     const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = this->value(points[i]);
}


template <int dim>
void
fill_injection_matrix(
  const DoFHandler<dim>                                 &coarse_dof_handler,
  const DoFHandler<dim>                                 &fine_dof_handler,
  SparsityPattern                                       &sparsity_pattern,
  SparseMatrix<double>                                  &transfer_matrix,
  const std::map<std::pair<types::global_cell_index, types::global_cell_index>,
                 std::vector<types::global_cell_index>> &parent_to_child_info,
  const std::vector<BoundingBox<dim>>                   &coarse_level_boxes,
  const std::vector<BoundingBox<dim>>                   &fine_level_boxes,
  const unsigned int                                     coarse_level)
{
  const FiniteElement<dim> &fe_dgq    = coarse_dof_handler.get_fe();
  const Triangulation<dim> &fine_tria = fine_dof_handler.get_triangulation();
  AffineConstraints<double> constraints;

  const std::vector<Point<dim>> &unit_support_points =
    fe_dgq.get_unit_support_points();

  DynamicSparsityPattern dsp;
  dsp.reinit(fine_dof_handler.n_dofs(), coarse_dof_handler.n_dofs());
  AffineConstraints<double>            dummy_constraints;
  std::vector<types::global_dof_index> coarse_dof_indices(
    fe_dgq.n_dofs_per_cell());
  std::vector<types::global_dof_index> fine_dof_indices(
    fe_dgq.n_dofs_per_cell());


  // Loop over coarse tria and print DoFs
  for (const auto &cell : coarse_dof_handler.active_cell_iterators())
    {
      cell->get_dof_indices(coarse_dof_indices);

      std::vector<types::global_dof_index> indices_of_children =
        parent_to_child_info.at({cell->active_cell_index(), coarse_level + 1});

      for (const auto &idx : indices_of_children)
        {
          DoFAccessor<dim, dim, dim, false> dof_accessor_child(
            &fine_tria, 0, idx, &fine_dof_handler);
          dof_accessor_child.get_dof_indices(fine_dof_indices);

          for (const types::global_dof_index row : fine_dof_indices)
            dsp.add_entries(row,
                            coarse_dof_indices.begin(),
                            coarse_dof_indices.end());
        }
    }

  // Filled sparsity pattern
  sparsity_pattern.copy_from(dsp);
  std::cout << "Sparsity pattern: filled" << std::endl;

  std::cout << "All level boxes[" << coarse_level
            << "] size: " << coarse_level_boxes.size() << std::endl;
  std::cout << "All level boxes[" << coarse_level + 1
            << "] size: " << fine_level_boxes.size() << std::endl;

  // Now onto filling the matrix...
  transfer_matrix.reinit(sparsity_pattern);
  const unsigned int dofs_per_cell = fe_dgq.n_dofs_per_cell();
  FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);

  for (const auto &cell : coarse_dof_handler.active_cell_iterators())
    {
      cell->get_dof_indices(coarse_dof_indices);

      const BoundingBox<dim> &coarse_box =
        coarse_level_boxes[cell->active_cell_index()];

      std::vector<types::global_dof_index> indices_of_children =
        parent_to_child_info.at({cell->active_cell_index(), coarse_level + 1});

      for (const auto &idx : indices_of_children)
        {
          DoFAccessor<dim, dim, dim, false> dof_accessor_child(
            &fine_tria, 0, idx, &fine_dof_handler);
          dof_accessor_child.get_dof_indices(fine_dof_indices);
          const BoundingBox<dim> &fine_bbox = fine_level_boxes[idx];

          local_matrix = 0.;

          // Now we plot the fine support points
          std::vector<Point<dim>> real_qpoints;
          real_qpoints.reserve(unit_support_points.size());
          for (const Point<dim> &p : unit_support_points)
            real_qpoints.push_back(fine_bbox.unit_to_real(p));

          for (unsigned int i = 0; i < coarse_dof_indices.size(); ++i)
            {
              const auto &p = coarse_box.real_to_unit(real_qpoints[i]);
              for (unsigned int j = 0; j < fine_dof_indices.size(); ++j)
                {
                  local_matrix(i, j) = fe_dgq.shape_value(j, p);
                }
            }

          constraints.distribute_local_to_global(local_matrix,
                                                 fine_dof_indices,
                                                 coarse_dof_indices,
                                                 transfer_matrix);
        }
    }
}



template <int dim>
class Poisson
{
private:
  void
  make_grid();
  void
  test_transfers();
  void
  assemble_system();
  void
  setup_multigrid();


  Triangulation<dim> tria;
#ifdef HEX
  MappingQ1<dim> mapping;
  FE_Q<dim>      fe_q;
#else
  MappingFE<dim>     mapping;
  FE_SimplexDGP<dim> fe_q;
#endif
  AffineConstraints<double>              constraints;
  SparsityPattern                        sparsity;
  DynamicSparsityPattern                 dsp;
  SparseMatrix<double>                   system_matrix;
  Vector<double>                         solution;
  Vector<double>                         system_rhs;
  std::unique_ptr<GridTools::Cache<dim>> cached_tria;
  std::unique_ptr<const Function<dim>>   rhs_function;
  std::unique_ptr<const Function<dim>>   analytical_solution;

public:
  Poisson(const GridType        &grid_type        = GridType::grid_generator,
          const PartitionerType &partitioner_type = PartitionerType::rtree,
          const SolutionType    &solution_type    = SolutionType::linear,
          const unsigned int                      = 0,
          const unsigned int fe_degree            = 1);
  void
  run();

  GridType        grid_type;
  PartitionerType partitioner_type;
  SolutionType    solution_type;
  unsigned int    extraction_level;

  DoFHandler<dim>                   original_dof_handler;
  std::vector<SparseMatrix<double>> injection_matrices;
  std::vector<SparsityPattern>      injection_sparsity_patterns;
};



// Helper function to create a dummy triangulation from BBoxes
template <int dim>
void
create_triangulation_from_bounding_boxes(
  Triangulation<dim>                  &dummy_tria,
  const std::vector<BoundingBox<dim>> &boxes)
{
  const unsigned int n_boxes          = boxes.size();
  const unsigned int vertices_per_box = (dim == 2) ? 4 : 8;

  // Pre-allocate space
  std::vector<Point<dim>>    vertices;
  std::vector<CellData<dim>> cells;
  vertices.reserve(n_boxes * vertices_per_box);
  cells.reserve(n_boxes);

  for (const auto &box : boxes)
    {
      const Point<dim> &p_min = box.get_boundary_points().first;
      const Point<dim> &p_max = box.get_boundary_points().second;

      // Each box gets its own set of vertices (allows overlaps!)
      const unsigned int vertex_offset = vertices.size();

      if constexpr (dim == 2)
        {
          // 4 corners of the bounding box
          vertices.push_back(Point<dim>(p_min[0], p_min[1])); // bottom-left
          vertices.push_back(Point<dim>(p_max[0], p_min[1])); // bottom-right
          vertices.push_back(Point<dim>(p_min[0], p_max[1])); // top-left
          vertices.push_back(Point<dim>(p_max[0], p_max[1])); // top-right

          // Create a quadrilateral cell from these 4 vertices
          CellData<dim> cell;
          cell.vertices[0] = vertex_offset + 0; // bottom-left
          cell.vertices[1] = vertex_offset + 1; // bottom-right
          cell.vertices[2] = vertex_offset + 2; // top-left
          cell.vertices[3] = vertex_offset + 3; // top-right
          cells.push_back(cell);
        }
      else if constexpr (dim == 3)
        {
          // 8 corners of the 3D bounding box
          vertices.push_back(Point<dim>(p_min[0], p_min[1], p_min[2]));
          vertices.push_back(Point<dim>(p_max[0], p_min[1], p_min[2]));
          vertices.push_back(Point<dim>(p_min[0], p_max[1], p_min[2]));
          vertices.push_back(Point<dim>(p_max[0], p_max[1], p_min[2]));
          vertices.push_back(Point<dim>(p_min[0], p_min[1], p_max[2]));
          vertices.push_back(Point<dim>(p_max[0], p_min[1], p_max[2]));
          vertices.push_back(Point<dim>(p_min[0], p_max[1], p_max[2]));
          vertices.push_back(Point<dim>(p_max[0], p_max[1], p_max[2]));

          // Create a hexahedral cell from these 8 vertices
          CellData<dim> cell;
          for (unsigned int v = 0; v < 8; ++v)
            cell.vertices[v] = vertex_offset + v;
          cells.push_back(cell);
        }
    }

  // Create the triangulation
  dummy_tria.create_triangulation(vertices, cells, SubCellData());
}



template <int dim>
Poisson<dim>::Poisson(const GridType        &grid_type,
                      const PartitionerType &partitioner_type,
                      const SolutionType    &solution_type,
                      const unsigned int     extraction_level,
                      const unsigned int     fe_degree)
  :
#ifdef HEX
  mapping()
#else
  mapping(FE_SimplexP<dim>{1})
#endif
  , fe_q(fe_degree)
  , grid_type(grid_type)
  , partitioner_type(partitioner_type)
  , solution_type(solution_type)
  , extraction_level(extraction_level)
  , original_dof_handler(tria)
{
  // Initialize manufactured solution
  if (solution_type == SolutionType::linear)
    analytical_solution = std::make_unique<SolutionLinear<dim>>();
  else if (solution_type == SolutionType::quadratic)
    analytical_solution = std::make_unique<SolutionQuadratic<dim>>();
  else if (solution_type == SolutionType::product)
    analytical_solution = std::make_unique<SolutionProduct<dim>>();
  else if (solution_type == SolutionType::product_sine)
    analytical_solution = std::make_unique<SolutionProductSine<dim>>();

  rhs_function = std::make_unique<const RightHandSide<dim>>(solution_type);
  constraints.close();
}

template <int dim>
void
Poisson<dim>::make_grid()
{
  GridIn<dim> grid_in;
  if (grid_type == GridType::unstructured)
    {
      if constexpr (dim == 2)
        {
          grid_in.attach_triangulation(tria);
#ifdef HEX
          std::ifstream gmsh_file(
            "../../meshes/t3.msh"); // unstructured square made by triangles
#else
          std::ifstream gmsh_file(
            "../../meshes/square_simplex_coarser.msh"); // unstructured square
                                                        // made by triangles
#endif
          grid_in.read_msh(gmsh_file);
          tria.refine_global(5); // 4
        }
      else if constexpr (dim == 3)
        {
          grid_in.attach_triangulation(tria);
#ifdef HEX
          std::ifstream filename("../../meshes/piston_3.inp"); // piston mesh
          grid_in.read_abaqus(filename);
#else
          std::ifstream filename(
            "../../meshes/gray_level_image1.vtk"); // liver or brain domain
          grid_in.read_vtk(filename);

#endif
        }
    }
  else
    {
#ifdef HEX
      GridGenerator::hyper_cube(tria, 0., 1.);
      tria.refine_global(8);
#else
      Triangulation<dim> tria_hex;
      GridGenerator::hyper_cube(tria_hex, 0., 1.);
      tria_hex.refine_global(3);
      GridGenerator::convert_hypercube_to_simplex_mesh(tria_hex, tria);
#endif
    }
  std::cout << "Size of tria: " << tria.n_active_cells() << std::endl;
  cached_tria = std::make_unique<GridTools::Cache<dim>>(tria, mapping);

  if (partitioner_type == PartitionerType::no_partition ||
      partitioner_type == PartitionerType::metis ||
      partitioner_type == PartitionerType::rtree)
    {
    }
  else
    {
      Assert(false, ExcMessage("Wrong partitioning."));
    }
}


// This function is used just to test stuff, it's kinda old, unused right now
template <int dim>
void
Poisson<dim>::test_transfers()
{
  if (partitioner_type == PartitionerType::rtree)
    {
      DoFHandler<dim> dof_handler(tria); // This is the finest DoF_Handler
      dof_handler.distribute_dofs(fe_q);

      namespace bgi = boost::geometry::index;
      static constexpr unsigned int max_elem_per_node =
        PolyUtils::constexpr_pow(2, dim + 1); // 2^dim
      static constexpr unsigned int min_elem_per_node = 4;

      std::vector<Point<dim>> support_points_vector(dof_handler.n_dofs());

      DoFTools::map_dofs_to_support_points(mapping,
                                           dof_handler,
                                           support_points_vector);

      auto tree =
        pack_rtree_of_indices<bgi::rstar<max_elem_per_node, min_elem_per_node>>(
          support_points_vector);

      std::cout << "Number of DoFs: " << dof_handler.n_dofs() << std::endl;
      std::cout << "Total number of available levels: " << n_levels(tree)
                << std::endl;
      std::vector<std::vector<BoundingBox<dim>>> all_level_boxes(
        n_levels(tree) + 1);

      static constexpr bool use_points = true;
      double                area       = 0.;

      for (unsigned int level_index = 0; level_index < n_levels(tree) + 1;
           ++level_index)
        {
          CellsAgglomerator<dim, decltype(tree), use_points> agglomerator{
            tree, level_index};
          const std::vector<std::vector<types::global_dof_index>> agglomerates =
            agglomerator.extract_agglomerates();
          // all_level_boxes[level_index].reserve(agglomerates.size());
          // std::cout << "agglomerates.size() = " << agglomerates.size()
          //           << " with indices:" << std::endl;
          for (const std::vector<types::global_dof_index> &agglo : agglomerates)
            {
              std::vector<Point<dim>> points_in_current_agglomerate;
              points_in_current_agglomerate.reserve(agglo.size());

              for (const auto &index : agglo)
                {
                  // std::cout << "Index " << index << " "
                  //           << " at point " << support_points_vector[index]
                  //           << "; ";
                  // std::cout << std::endl;
                  points_in_current_agglomerate.push_back(
                    support_points_vector[index]);
                }
              // std::cout << std::endl;

              BoundingBox<dim> bbox{points_in_current_agglomerate};
              all_level_boxes[level_index].emplace_back(
                points_in_current_agglomerate);

              area += bbox.volume();
              AssertThrow(bbox.volume() > 1e-10,
                          ExcMessage("Box too small..."));
            }

          std::cout << "Level " << level_index
                    << " has following number of agglomerates: "
                    << agglomerates.size() << std::endl;

          // std::cout << "Total area covered by agglomerates: " << area
          //           << std::endl;
        }


      unsigned int       my_level = 1;
      Triangulation<dim> tria_bbox;
      create_triangulation_from_bounding_boxes(tria_bbox,
                                               all_level_boxes[my_level]);

      Triangulation<dim> tria_bbox_child;
      create_triangulation_from_bounding_boxes(tria_bbox_child,
                                               all_level_boxes[my_level + 1]);

      // for (const auto &cell : tria_bbox.active_cell_iterators())
      //   {
      //     std::cout << "Cell with index " << cell->active_cell_index()
      //               << " has vertices: ";
      //     for (unsigned int v = 0; v < 4; ++v)
      //       std::cout << cell->vertex(v) << " ";
      //     std::cout << "to be compared with bbox: ("
      //               << all_level_boxes[my_level][cell->active_cell_index()]
      //                    .get_boundary_points()
      //                    .first
      //               << " , "
      //               << all_level_boxes[my_level][cell->active_cell_index()]
      //                    .get_boundary_points()
      //                    .second
      //               << ")" << std::endl;


      //     std::cout << std::endl;
      //   }

      CellsAgglomerator<dim, decltype(tree), use_points> agglomerator{tree,
                                                                      my_level};
      agglomerator.extract_agglomerates();
      const std::map<
        std::pair<types::global_cell_index, types::global_cell_index>,
        std::vector<types::global_cell_index>> &parent_to_child_info =
        agglomerator.get_hierarchy();
      // for (const auto &[key, value] : parent_to_child_info)
      //   {
      //     std::cout << "We are on level " << key.second << std::endl;

      //     std::cout << "Parent cell " << key.first << " has children: ";
      //     for (const types::global_dof_index child_index : value)
      //       {
      //         std::cout << child_index << " ";
      //       }
      //     std::cout << std::endl;
      //   }


      // DoFs
      FE_DGQ<dim>     fe_dgq(fe_q.degree);
      DoFHandler<dim> coarse_dof_handler(tria_bbox);
      coarse_dof_handler.distribute_dofs(fe_dgq);
      std::vector<types::global_dof_index> dof_indices(
        fe_dgq.n_dofs_per_cell());

      // child
      DoFHandler<dim> coarse_dof_handler_child(tria_bbox_child);
      coarse_dof_handler_child.distribute_dofs(fe_dgq);
      std::vector<types::global_dof_index> dof_indices_child(
        fe_dgq.n_dofs_per_cell());

      const std::vector<Point<dim>> &unit_support_points =
        fe_dgq.get_unit_support_points();

      // Loop over coarse tria and print DoFs
      // for (const auto &cell : coarse_dof_handler.active_cell_iterators())
      //   {
      //     std::cout << "Coarse cell (which is already a box) "
      //               << cell->active_cell_index() << " has DoFs: ";
      //     cell->get_dof_indices(dof_indices);
      //     for (const auto &dof_index : dof_indices)
      //       std::cout << dof_index << " ";
      //     std::cout << std::endl;

      //     const BoundingBox<dim> &coarse_box =
      //       all_level_boxes[my_level][cell->active_cell_index()];
      //     std::cout << "Coarse box has boundary points: "
      //               << coarse_box.get_boundary_points().first << " , "
      //               << coarse_box.get_boundary_points().second << std::endl;

      //     std::vector<types::global_dof_index> indices_of_children =
      //       parent_to_child_info.at({cell->active_cell_index(), my_level});

      //     for (const auto &idx : indices_of_children)
      //       {
      //         DoFAccessor<dim, dim, dim, false> dof_accessor_child(
      //           &tria_bbox_child, 0, idx, &coarse_dof_handler_child);

      //         std::cout << "And here are the DoF indices of child " << idx
      //                   << ": ";
      //         dof_accessor_child.get_dof_indices(dof_indices_child);
      //         for (const auto &dof_index_child : dof_indices_child)
      //           std::cout << dof_index_child << " ";
      //         std::cout << std::endl;

      //         const BoundingBox<dim> &fine_bbox =
      //           all_level_boxes[my_level + 1][idx];
      //         std::cout << "Children box " << idx << " has boundary points: "
      //                   << fine_bbox.get_boundary_points().first << " , "
      //                   << fine_bbox.get_boundary_points().second <<
      //                   std::endl;


      //         // Now we plot the fine support points
      //         std::vector<Point<dim>> real_qpoints;
      //         real_qpoints.reserve(unit_support_points.size());
      //         for (const Point<dim> &p : unit_support_points)
      //           {
      //             std::cout
      //               << "Fine support point: " << fine_bbox.unit_to_real(p)
      //               << std::endl;
      //             real_qpoints.push_back(fine_bbox.unit_to_real(p));

      //             // Let's try to evaluate
      //             unsigned int     basis_idx = 0;
      //             const Point<dim> p_mapped =
      //               coarse_box.real_to_unit(fine_bbox.unit_to_real(p));

      //             std::cout << " Eval at mapped point " << p_mapped << " : "
      //                       << fe_dgq.shape_value(basis_idx, p_mapped)
      //                       << std::endl;
      //           }
      //       }
      //   }

      // Test we can build a transfer matrix P

      SparsityPattern        sparsity_pattern;
      DynamicSparsityPattern dsp;
      dsp.reinit(coarse_dof_handler_child.n_dofs(),
                 coarse_dof_handler.n_dofs());
      AffineConstraints<double>            dummy_constraints;
      std::vector<types::global_dof_index> coarse_dof_indices(
        fe_dgq.n_dofs_per_cell());
      std::vector<types::global_dof_index> fine_dof_indices(
        fe_dgq.n_dofs_per_cell());

      // Loop over coarse tria and store DoFs
      for (const auto &cell : coarse_dof_handler.active_cell_iterators())
        {
          cell->get_dof_indices(coarse_dof_indices);

          std::vector<types::global_dof_index> indices_of_children =
            parent_to_child_info.at({cell->active_cell_index(), my_level});

          for (const auto &idx : indices_of_children)
            {
              DoFAccessor<dim, dim, dim, false> dof_accessor_child(
                &tria_bbox_child, 0, idx, &coarse_dof_handler_child);
              dof_accessor_child.get_dof_indices(fine_dof_indices);

              for (const types::global_dof_index row : fine_dof_indices)
                dsp.add_entries(row,
                                coarse_dof_indices.begin(),
                                coarse_dof_indices.end());
            }
        }

      // Filled sparsity pattern
      sparsity_pattern.copy_from(dsp);
      std::cout << "Filled sparsity pattern" << std::endl;

      // Now onto filling the matrix...
      SparseMatrix<double> transfer_matrix;
      transfer_matrix.reinit(sparsity_pattern);
      const unsigned int dofs_per_cell = fe_dgq.n_dofs_per_cell();
      FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);

      for (const auto &cell : coarse_dof_handler.active_cell_iterators())
        {
          cell->get_dof_indices(coarse_dof_indices);

          const BoundingBox<dim> &coarse_box =
            all_level_boxes[my_level][cell->active_cell_index()];

          std::vector<types::global_dof_index> indices_of_children =
            parent_to_child_info.at({cell->active_cell_index(), my_level});

          for (const auto &idx : indices_of_children)
            {
              DoFAccessor<dim, dim, dim, false> dof_accessor_child(
                &tria_bbox_child, 0, idx, &coarse_dof_handler_child);
              dof_accessor_child.get_dof_indices(fine_dof_indices);

              const BoundingBox<dim> &fine_bbox =
                all_level_boxes[my_level + 1][idx];

              local_matrix = 0.;

              std::vector<Point<dim>> real_qpoints;
              real_qpoints.reserve(unit_support_points.size());
              for (const Point<dim> &p : unit_support_points)
                real_qpoints.push_back(fine_bbox.unit_to_real(p));

              for (unsigned int i = 0; i < coarse_dof_indices.size(); ++i)
                {
                  const auto &p = coarse_box.real_to_unit(real_qpoints[i]);
                  for (unsigned int j = 0; j < fine_dof_indices.size(); ++j)
                    {
                      local_matrix(i, j) = fe_dgq.shape_value(j, p);
                    }
                }

              constraints.distribute_local_to_global(local_matrix,
                                                     fine_dof_indices,
                                                     coarse_dof_indices,
                                                     transfer_matrix);
            }
        }


      std::cout << "Built transfer matrix with dimensions "
                << transfer_matrix.m() << " x " << transfer_matrix.n()
                << std::endl;
      // std::string filename_tr =
      //   std::string("transfer_matrix_agglo_to_agglo.txt");
      // std::ofstream outfile_tr(filename_tr);
      // transfer_matrix.print_as_numpy_arrays(outfile_tr);
      // outfile_tr.close();

      std::cout
        << "Now let's build the transfer from original tria to (finest) agglomerated tria"
        << std::endl;


      CellsAgglomerator<dim, decltype(tree), use_points> agglomerator_test{
        tree, my_level + 1};
      const std::vector<std::vector<types::global_dof_index>> agglomerates =
        agglomerator_test.extract_agglomerates();

      std::vector<types::global_dof_index> dof_indices_agglo_tria(
        fe_dgq.n_dofs_per_cell());

      original_dof_handler.distribute_dofs(
        fe_q); //! original dof handler has to be distributed

      SparsityPattern        sparsity_pattern_agglo_to_original_tria;
      DynamicSparsityPattern dsp_agglo_to_original_tria;
      dsp_agglo_to_original_tria.reinit(original_dof_handler.n_dofs(),
                                        coarse_dof_handler_child.n_dofs());

      unsigned int agglo_index = 0;
      for (const auto &cell : coarse_dof_handler_child.active_cell_iterators())
        {
          cell->get_dof_indices(dof_indices_agglo_tria);

          for (const types::global_dof_index dof_idx :
               agglomerates[agglo_index])
            dsp_agglo_to_original_tria.add_entries(
              dof_idx,
              dof_indices_agglo_tria.begin(),
              dof_indices_agglo_tria.end());

          ++agglo_index;
        }

      std::cout << "Done sparsity agglo to original fine tria" << std::endl;

      // Now onto the matrix...
      SparseMatrix<double> transfer_matrix_agglo_to_original_tria;
      sparsity_pattern_agglo_to_original_tria.copy_from(
        dsp_agglo_to_original_tria);
      transfer_matrix_agglo_to_original_tria.reinit(
        sparsity_pattern_agglo_to_original_tria);

      // reset agglo_index
      agglo_index = 0;
      for (const auto &cell : coarse_dof_handler_child.active_cell_iterators())
        {
          // Extract the bounding box, using the index
          // std::cout << "Cell with index " << cell->active_cell_index()
          //           << " has vertices: ";
          // for (unsigned int v = 0; v < 4; ++v)
          //   std::cout << cell->vertex(v) << " ";
          // std::cout << std::endl;

          cell->get_dof_indices(dof_indices_agglo_tria);

          const BoundingBox<dim> &coarse_box =
            all_level_boxes[my_level + 1][cell->active_cell_index()];

          // Now I want to retrieve the fine support points and indices
          // std::cout << "Showing FINE indices for agglomerate " << agglo_index
          //           << std::endl;
          // std::cout << "The current box is "
          //           << coarse_box.get_boundary_points().first << " , "
          //           << coarse_box.get_boundary_points().second << std::endl;

          const unsigned int n_fine_support_points =
            agglomerates[agglo_index].size();
          // std::cout << "Number of support points we have to evaluate: "
          //           << n_fine_support_points << std::endl;

          const std::vector<types::global_dof_index> fine_indices =
            agglomerates[agglo_index];

          // for (const types::global_dof_index index : fine_indices)
          //   {
          //     std::cout << "Fine DoF Index " << index << " "
          //               << "at (fine) support point "
          //               << support_points_vector[index] << "; ";
          //     std::cout << std::endl;
          //   }

          FullMatrix<double> local_matrix2(n_fine_support_points,
                                           fe_dgq.n_dofs_per_cell());
          local_matrix2 = 0.;

          for (unsigned int i = 0; i < n_fine_support_points; ++i)
            {
              const Point<dim> p =
                coarse_box.real_to_unit(support_points_vector[fine_indices[i]]);
              for (unsigned int j = 0; j < dof_indices_agglo_tria.size(); ++j)
                {
                  local_matrix2(i, j) = fe_dgq.shape_value(j, p);
                }
            }

          constraints.distribute_local_to_global(
            local_matrix2,
            fine_indices,           // tria original
            dof_indices_agglo_tria, // agglomerated tria
            transfer_matrix_agglo_to_original_tria);

          ++agglo_index; // advance to next agglomerate
          // std::cout << std::endl;
        }

      std::cout
        << "Built transfer matrix agglo to original tria with dimensions "
        << transfer_matrix_agglo_to_original_tria.m() << " x "
        << transfer_matrix_agglo_to_original_tria.n() << std::endl;

      // std::string filename =
      //   std::string("transfer_matrix_agglo_to_original_tria.txt");
      // std::ofstream outfile(filename);
      // transfer_matrix_agglo_to_original_tria.print_as_numpy_arrays(outfile);
      // outfile.close();
    }
}



template <int dim>
void
Poisson<dim>::assemble_system()
{
  original_dof_handler.distribute_dofs(fe_q); //! done before

  // assembling standard Poisson system
  dsp.reinit(original_dof_handler.n_dofs(), original_dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(original_dof_handler, dsp);
  sparsity.copy_from(dsp);
  system_matrix.reinit(sparsity);

  solution.reinit(original_dof_handler.n_dofs());
  system_rhs.reinit(original_dof_handler.n_dofs());

  const QGauss<dim> quadrature_formula(fe_q.get_degree() + 1);
  FEValues<dim>     fe_values(mapping,
                          fe_q,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe_q.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : original_dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);

      cell_matrix = 0;
      cell_rhs    = 0;

      // Evaluate RHS function at all quadrature points
      const unsigned int  n_q_points = fe_values.n_quadrature_points;
      std::vector<double> rhs_values(n_q_points);
      rhs_function->value_list(fe_values.get_quadrature_points(), rhs_values);

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int j : fe_values.dof_indices())
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                 fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                 fe_values.JxW(q_index));           // dx

          for (const unsigned int i : fe_values.dof_indices())
            cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            rhs_values[q_index] *               // f(x_q)
                            fe_values.JxW(q_index));            // dx
        }
      cell->get_dof_indices(local_dof_indices);

      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
          system_matrix.add(local_dof_indices[i],
                            local_dof_indices[j],
                            cell_matrix(i, j));

      for (const unsigned int i : fe_values.dof_indices())
        system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }


  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(original_dof_handler,
                                           types::boundary_id(0),
                                           *analytical_solution,
                                           boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution,
                                     system_rhs);

  std::cout << "Built finest system matrix with dimensions "
            << system_matrix.m() << " x " << system_matrix.n() << std::endl;
  std::string   filename = std::string("system_matrix.txt");
  std::ofstream outfile(filename);
  system_matrix.print_as_numpy_arrays(outfile);
  outfile.close();

  {
    // Let's print the fine triangulation
    GridOut       grid_out;
    std::ofstream out("fine_tria.vtk");
    grid_out.write_vtk(tria, out);
  }
}



// WIP
template <int dim>
void
Poisson<dim>::setup_multigrid()
{
  // Setup rtree with support points and modify the bboxes
  namespace bgi                                   = boost::geometry::index;
  static constexpr unsigned int max_elem_per_node = 16;
  // PolyUtils::constexpr_pow(2, dim + 1); // 2^dim
  static constexpr unsigned int min_elem_per_node = 8;
  static constexpr bool         use_points        = true;
  FE_DGQ<dim>                   fe_dg(fe_q.get_degree());

  std::vector<Point<dim>> support_points_vector(original_dof_handler.n_dofs());

  DoFTools::map_dofs_to_support_points(mapping,
                                       original_dof_handler,
                                       support_points_vector);

  auto tree =
    pack_rtree_of_indices<bgi::rstar<max_elem_per_node, min_elem_per_node>>(
      support_points_vector);
  std::cout << "======================= Multigrid testing ==================="
            << std::endl;

  std::cout << "Number of levels in the tree: " << n_levels(tree) << std::endl;

  std::vector<std::vector<BoundingBox<dim>>> all_level_boxes(
    n_levels(tree)); // N. B. there is an off by 1. all_level_boxes[0] =
                     // boxes at level 1  of the tree

  // This cycle creates all the bounding boxes at each level
  for (unsigned int i = 0; i < n_levels(tree); ++i)
    {
      CellsAgglomerator<dim, decltype(tree), use_points>      agglomerator{tree,
                                                                      i + 1};
      const std::vector<std::vector<types::global_dof_index>> agglomerates =
        agglomerator.extract_agglomerates();
      all_level_boxes[i].reserve(agglomerates.size());

      for (const std::vector<types::global_dof_index> &agglo : agglomerates)
        {
          std::vector<Point<dim>> points_in_current_agglomerate;
          points_in_current_agglomerate.reserve(agglo.size());

          for (const auto &index : agglo)
            points_in_current_agglomerate.push_back(
              support_points_vector[index]);

          BoundingBox<dim> bbox{points_in_current_agglomerate};
          all_level_boxes[i].emplace_back(points_in_current_agglomerate);
        }

    } // Bbox creator for each level


  std::cout << "Finished creating bounding boxes for multigrid levels"
            << std::endl;

  std::vector<std::unique_ptr<Triangulation<dim>>> triangulations;
  triangulations.reserve(n_levels(tree));

  std::vector<std::unique_ptr<DoFHandler<dim>>> all_level_support_DoFHandlers;
  all_level_support_DoFHandlers.reserve(n_levels(tree));


  for (unsigned int i = 0; i < n_levels(tree); ++i)
    {
      triangulations.push_back(std::make_unique<Triangulation<dim>>());
      create_triangulation_from_bounding_boxes(*triangulations[i],
                                               all_level_boxes[i]);

      all_level_support_DoFHandlers.push_back(
        std::make_unique<DoFHandler<dim>>(*triangulations[i]));
      all_level_support_DoFHandlers[i]->distribute_dofs(fe_dg);

      std::cout << "Created tria with "
                << all_level_support_DoFHandlers[i]->n_dofs() << " DoFs "
                << std::endl;
    }
  std::cout << "While the finest tria has " << original_dof_handler.n_dofs()
            << " DoFs." << std::endl;

  // Check that sizes of the data structures are consistent
  AssertThrow(all_level_support_DoFHandlers.size() == n_levels(tree),
              ExcMessage(
                "Inconsistent number of DoFHandlers for multigrid levels"));

  // Output the Bboxes trias at each level
  std::cout << "Outputting bounding box trias at each level" << std::endl;
  for (unsigned int level = 0; level < n_levels(tree); ++level)
    {
      GridOut       grid_out;
      std::ofstream out("bboxes_level_" + std::to_string(level) + ".vtk");
      grid_out.write_vtk(*triangulations[level], out);

      std::cout << "h_min at level " << level << " is "
                << GridTools::minimal_cell_diameter(*triangulations[level])
                << std::endl;
      std::cout << "h_max at level " << level << " is "
                << GridTools::maximal_cell_diameter(*triangulations[level])
                << std::endl;
    }

  std::cout << "h_min at level " << n_levels(tree) << " is "
            << GridTools::minimal_cell_diameter(tria) << std::endl;
  std::cout << "h_max at level " << n_levels(tree) << " is "
            << GridTools::maximal_cell_diameter(tria) << std::endl;

  injection_matrices.resize(n_levels(tree));
  injection_sparsity_patterns.resize(n_levels(tree));

  for (unsigned int level = 0; level < n_levels(tree) - 1; ++level)
    {
      CellsAgglomerator<dim, decltype(tree), use_points> agglomerator{tree,
                                                                      level +
                                                                        1};
      agglomerator.extract_agglomerates();
      const std::map<
        std::pair<types::global_cell_index, types::global_cell_index>,
        std::vector<types::global_cell_index>> &parent_to_child_info =
        agglomerator.get_hierarchy();

      fill_injection_matrix<dim>(*all_level_support_DoFHandlers[level],
                                 *all_level_support_DoFHandlers[level + 1],
                                 injection_sparsity_patterns[level],
                                 injection_matrices[level],
                                 parent_to_child_info,
                                 all_level_boxes[level],
                                 all_level_boxes[level + 1],
                                 level /*coarse_level*/);

      std::cout << "Built transfer matrix with size "
                << injection_matrices[level].m() << " x "
                << injection_matrices[level].n() << " from level " << level + 1
                << " to level " << level + 2 << std::endl;
    }

  // Now fill the last injection from finest level to original tria
  {
    CellsAgglomerator<dim, decltype(tree), use_points> agglomerator_test{
      tree, n_levels(tree)};
    const std::vector<std::vector<types::global_dof_index>> agglomerates =
      agglomerator_test.extract_agglomerates();

    std::vector<types::global_dof_index> dof_indices_agglo_tria(
      fe_dg.n_dofs_per_cell());

    DynamicSparsityPattern dsp_agglo_to_original_tria;
    dsp_agglo_to_original_tria.reinit(
      original_dof_handler.n_dofs(),
      all_level_support_DoFHandlers[n_levels(tree) - 1]->n_dofs());

    unsigned int agglo_index = 0;
    for (const auto &cell : all_level_support_DoFHandlers[n_levels(tree) - 1]
                              ->active_cell_iterators())
      {
        cell->get_dof_indices(dof_indices_agglo_tria);

        for (const types::global_dof_index dof_idx : agglomerates[agglo_index])
          dsp_agglo_to_original_tria.add_entries(dof_idx,
                                                 dof_indices_agglo_tria.begin(),
                                                 dof_indices_agglo_tria.end());

        ++agglo_index;
      }

    std::cout << "Done sparsity agglo to original fine tria" << std::endl;

    // Now build the matrix...
    injection_sparsity_patterns[n_levels(tree) - 1].copy_from(
      dsp_agglo_to_original_tria);
    injection_matrices[n_levels(tree) - 1].reinit(
      injection_sparsity_patterns[n_levels(tree) - 1]);

    // reset agglo_index
    agglo_index = 0;
    for (const auto &cell : all_level_support_DoFHandlers[n_levels(tree) - 1]
                              ->active_cell_iterators())
      {
        cell->get_dof_indices(dof_indices_agglo_tria);

        const BoundingBox<dim> &coarse_box =
          all_level_boxes[n_levels(tree) - 1][cell->active_cell_index()];

        const unsigned int n_fine_support_points =
          agglomerates[agglo_index].size();

        const std::vector<types::global_dof_index> fine_indices =
          agglomerates[agglo_index];


        FullMatrix<double> local_matrix2(n_fine_support_points,
                                         fe_dg.n_dofs_per_cell());
        local_matrix2 = 0.;

        for (unsigned int i = 0; i < n_fine_support_points; ++i)
          {
            const Point<dim> p =
              coarse_box.real_to_unit(support_points_vector[fine_indices[i]]);
            for (unsigned int j = 0; j < dof_indices_agglo_tria.size(); ++j)
              {
                local_matrix2(i, j) = fe_dg.shape_value(j, p);
              }
          }

        constraints.distribute_local_to_global(
          local_matrix2,
          fine_indices,           // tria original
          dof_indices_agglo_tria, // agglomerated tria
          injection_matrices[n_levels(tree) - 1]);

        ++agglo_index; // advance to next agglomerate
      }

    std::cout << "Built transfer matrix agglo to original tria with dimensions "
              << injection_matrices[n_levels(tree) - 1].m() << " x "
              << injection_matrices[n_levels(tree) - 1].n() << std::endl;
  }


  std::cout << "Finished setting up multigrid transfer operators" << std::endl;

  // Output all transfer matrices for numpy
  for (unsigned int level = 0; level < n_levels(tree); ++level)
    {
      std::string filename_tr =
        std::string("transfer_matrix_level_") +
        Utilities::int_to_string(level) + std::string("_to_") +
        Utilities::int_to_string(level + 1) + std::string(".txt");
      std::ofstream outfile_tr(filename_tr);
      injection_matrices[level].print_as_numpy_arrays(outfile_tr);
      outfile_tr.close();
    }

  // Let's add an output for paraview printing some basis functions. The output
  // should be on the original fine tria
  std::cout << "Output some basis functions on the original fine tria"
            << std::endl;
  {
    for (unsigned int level = 0; level < n_levels(tree); ++level)
      {
        for (unsigned int shape_fun_idx = 0; shape_fun_idx < 4; ++shape_fun_idx)
          {
            Vector<double> source_shape_fun(injection_matrices[level].n());
            source_shape_fun[shape_fun_idx] = 1.0;

            for (unsigned int inner_level = level; inner_level < n_levels(tree);
                 ++inner_level)
              {
                Vector<double> target_shape_fun(
                  injection_matrices[inner_level].m());

                injection_matrices[inner_level].vmult(target_shape_fun,
                                                      source_shape_fun);

                source_shape_fun.reinit(injection_matrices[inner_level].m());
                source_shape_fun = target_shape_fun;
              }

            {
              DataOut<dim> data_out;
              data_out.attach_dof_handler(original_dof_handler);

              data_out.add_data_vector(
                source_shape_fun,
                "shape_function_level_" + Utilities::int_to_string(level) +
                  "_idx_" + Utilities::int_to_string(shape_fun_idx),
                DataOut<dim>::type_dof_data);

              data_out.build_patches(mapping);

              const std::string filename =
                "basis_function_level_" + Utilities::int_to_string(level) +
                "_idx_" + Utilities::int_to_string(shape_fun_idx) + ".vtu";
              std::ofstream output(filename);
              data_out.write_vtu(output);
            }
          }
      }
  }

  std::cout << "Support points vector size original tria: "
            << support_points_vector.size() << std::endl;

  for (const auto &mat : injection_matrices)
    std::cout << "Injection matrix size: " << mat.m() << " x " << mat.n()
              << std::endl;

  std::vector<TrilinosWrappers::SparseMatrix> trilinos_transfer_matrices(
    n_levels(tree));

  // Copy everything to Trilinos matrices to use already existing stuff
  for (unsigned int level = 0; level < n_levels(tree); ++level)
    {
      trilinos_transfer_matrices[level].reinit(injection_matrices[level]);
    }

  AmgProjector<dim, TrilinosWrappers::SparseMatrix, double> amg_projector(
    trilinos_transfer_matrices); // Initialize projector
  std::cout << "Initialized AMG projector" << std::endl;

  MGLevelObject<std::unique_ptr<TrilinosWrappers::SparseMatrix>>
    multigrid_matrices(0, n_levels(tree));

  multigrid_matrices[multigrid_matrices.max_level()] =
    std::make_unique<TrilinosWrappers::SparseMatrix>();

  // Set up finest level system matrix (copy the matrix content)
  multigrid_matrices[multigrid_matrices.max_level()]->reinit(system_matrix);
  std::cout << "Built finest operator" << std::endl;

  amg_projector.compute_level_matrices(multigrid_matrices);
  std::cout << "Projected using transfer_matrices:" << std::endl;

  std::cout << "Check dimensions of level operators" << std::endl;
  for (unsigned int level = 0; level <= multigrid_matrices.max_level(); ++level)
    std::cout << "Level " << level
              << " operator size: " << multigrid_matrices[level]->m() << " x "
              << multigrid_matrices[level]->n() << std::endl;

  // Setup multigrid

  // Multigrid matrices
  using LevelMatrixType = TrilinosWrappers::SparseMatrix;
  using VectorType      = LinearAlgebra::distributed::Vector<double>;
  mg::Matrix<VectorType> mg_matrix(multigrid_matrices);

  using SmootherType = PreconditionChebyshev<LevelMatrixType, VectorType>;
  mg::SmootherRelaxation<SmootherType, VectorType>     mg_smoother;
  MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
  smoother_data.resize(0, n_levels(tree) + 1);

  std::cout << "Setting up smoothers" << std::endl;
  std::cout << "Setting up finest level smoother at level " << n_levels(tree)
            << std::endl;

  VectorType diag_inverse(system_matrix.m());
  for (unsigned int row = 0; row < system_matrix.m(); ++row)
    diag_inverse[row] = 1. / system_matrix.diag_element(row);
  diag_inverse.compress(VectorOperation::insert);

  std::vector<VectorType> diag_inverses(n_levels(tree) + 1);
  diag_inverses[n_levels(tree)] = diag_inverse;

  smoother_data[n_levels(tree)].preconditioner =
    std::make_shared<DiagonalMatrix<VectorType>>(diag_inverses[n_levels(tree)]);


  for (unsigned int level = 0; level < n_levels(tree); ++level)
    {
      // For simplicity using the same degree for all levels
      smoother_data[level].smoothing_range = 8;
      diag_inverses[level].reinit(
        multigrid_matrices[level]->m()); // need to reinit
      for (unsigned int row = 0; row < multigrid_matrices[level]->m(); ++row)
        diag_inverses[level][row] =
          1. / multigrid_matrices[level]->diag_element(row);
      diag_inverses[level].compress(VectorOperation::insert);

      smoother_data[level].preconditioner =
        std::make_shared<DiagonalMatrix<VectorType>>(diag_inverses[level]);

      std::cout << "Level " << level << " smoother set up " << std::endl;
    }

  std::cout << "Initialized smoothers data" << std::endl;

  for (unsigned int level = 0; level < n_levels(tree) + 1; ++level)
    {
      if (level > 0)
        {
          smoother_data[level].smoothing_range     = 20.; // 15.;
          smoother_data[level].degree              = 5;   // 5;
          smoother_data[level].eig_cg_n_iterations = 20;
        }
      else
        {
          smoother_data[0].smoothing_range = 1e-3;
          smoother_data[0].degree = 5; // numbers::invalid_unsigned_int;
          smoother_data[0].eig_cg_n_iterations = 20;
        }
    }

  mg_smoother.set_steps(10);
  mg_smoother.initialize(multigrid_matrices, smoother_data);

  std::cout << "Initialized  smoothers" << std::endl;

  // Define coarse grid solver
  const unsigned int min_level = 0;
  Utils::MGCoarseDirect<VectorType,
                        TrilinosWrappers::SparseMatrix,
                        TrilinosWrappers::SolverDirect>
    mg_coarse(*multigrid_matrices[min_level]);

  // Transfers
  MGLevelObject<TrilinosWrappers::SparseMatrix *> mg_level_transfers(
    0, n_levels(tree));
  for (unsigned int l = 0; l < n_levels(tree); ++l)
    mg_level_transfers[l] = &trilinos_transfer_matrices[l];

  std::vector<DoFHandler<dim> *> dof_handlers(n_levels(tree) + 1);
  for (unsigned int l = 0; l < dof_handlers.size() - 1; ++l)
    dof_handlers[l] = all_level_support_DoFHandlers[l].get();
  dof_handlers[n_levels(tree)] = &original_dof_handler;

  unsigned int lev = 0;
  for (const auto &dh : dof_handlers)
    std::cout << "Number of DoFs in level " << lev++ << ": " << dh->n_dofs()
              << std::endl;

  MGTransferAgglomeration<dim, VectorType> mg_transfer(mg_level_transfers,
                                                       dof_handlers);
  std::cout << "MG transfers initialized" << std::endl;

  // Define multigrid object and convert to preconditioner.
  Multigrid<VectorType> mg(mg_matrix,
                           mg_coarse,
                           mg_transfer,
                           mg_smoother,
                           mg_smoother,
                           min_level,
                           numbers::invalid_unsigned_int,
                           Multigrid<VectorType>::v_cycle);

  PreconditionMG<dim, VectorType, MGTransferAgglomeration<dim, VectorType>>
    preconditioner(original_dof_handler, mg, mg_transfer);

  VectorType dist_solution;
  VectorType dist_rhs;
  dist_solution.reinit(original_dof_handler.n_dofs());
  dist_rhs.reinit(original_dof_handler.n_dofs());
  for (unsigned int i = 0; i < system_rhs.size(); ++i)
    dist_rhs[i] = system_rhs[i];
  dist_rhs.compress(VectorOperation::insert);
  ReductionControl solver_control(10000, 1e-9, 1e-6, true, true);
  // SolverControl        solver_control(1000, 1e-9, true, true);
  SolverCG<VectorType> cg(solver_control);
  double               start, stop;
  std::cout << "Start solver" << std::endl;
  start = MPI_Wtime();
  cg.solve(system_matrix, dist_solution, dist_rhs, preconditioner);
  stop = MPI_Wtime();
  std::cout << "Agglo AMG elapsed time: " << stop - start << "[s]" << std::endl;

  std::cout << "Initial value: " << solver_control.initial_value() << std::endl;
  std::cout << "Converged in " << solver_control.last_step()
            << " iterations with value " << solver_control.last_value()
            << std::endl;

  [[maybe_unused]] auto output_results = [&]() -> void {
    std::cout << "Output results" << std::endl;
    DataOut<dim> data_out;
    data_out.attach_dof_handler(original_dof_handler);
    data_out.add_data_vector(dist_solution,
                             "interpolated_solution",
                             DataOut<dim>::type_dof_data);

    Vector<float> subdomain(tria.n_active_cells());

    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = tria.locally_owned_subdomain();

    data_out.add_data_vector(subdomain, "subdomain");

    Vector<float> agglo_idx(tria.n_active_cells());
    for (const auto &cell : tria.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          agglo_idx[cell->active_cell_index()] = cell->material_id();
      }
    data_out.add_data_vector(agglo_idx,
                             "agglo_idx",
                             DataOut<dim>::type_cell_data);

    data_out.build_patches(mapping);
    const std::string filename = ("agglo_mg." + Utilities::int_to_string(1, 4));
    std::ofstream     output((filename + ".vtu").c_str());
    data_out.write_vtu(output);

    {
      std::vector<std::string> filenames;
      for (unsigned int i = 0;
           i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
           i++)
        {
          filenames.push_back("agglo_mg." + Utilities::int_to_string(i, 4) +
                              ".vtu");
        }
      std::ofstream master_output("agglo_mg.pvtu");
      data_out.write_pvtu_record(master_output, filenames);
    }
  };

  if (original_dof_handler.n_dofs() < 3e6)
    output_results();
}



template <int dim>
void
Poisson<dim>::run()
{
  make_grid();
  // test_transfers();
  auto start = std::chrono::high_resolution_clock::now();
  assemble_system();
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
    std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  std::cout << "Time taken by assemble_system(): " << duration.count() / 1e6
            << " seconds" << std::endl;

  setup_multigrid();
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  deallog.depth_console(10);
  {
#ifdef HEX
    for (unsigned int fe_degree : {1})
#else
    for (unsigned int fe_degree : {1, 2, 3})
#endif
      {
        std::cout << "Fe degree: " << fe_degree << std::endl;
        Poisson<2> poisson_problem{
          GridType::grid_generator, // GridType::grid_generator
          PartitionerType::rtree,
          SolutionType::product_sine,
          3 /*extraction_level using 3 now*/,
          fe_degree};
        poisson_problem.run();
      }
  }
  std::cout << std::endl;
  return 0;
}
