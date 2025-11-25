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

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <agglomeration_handler.h>
#include <fe_agglodgp.h>
#include <poly_utils.h>

#include <algorithm>
#include <chrono>

#define HEX TRUE

struct ConvergenceInfo
{
  ConvergenceInfo() = default;
  void
  add(const std::pair<types::global_dof_index, std::pair<double, double>>
        &dofs_and_errs)
  {
    vec_data.push_back(dofs_and_errs);
  }



  void
  print()
  {
    Assert(vec_data.size() > 0, ExcInternalError());
    for (const auto &dof_and_errs : vec_data)
      std::cout << std::left << std::setw(24) << std::scientific
                << "N DoFs: " << dof_and_errs.first << std::endl;

    for (const auto &dof_and_errs : vec_data)
      std::cout << std::left << std::setw(24) << std::scientific
                << "L2 error: " << dof_and_errs.second.first << std::endl;
    for (const auto &dof_and_errs : vec_data)
      std::cout << std::left << std::setw(24) << std::scientific
                << "H1 error: " << dof_and_errs.second.second << std::endl;
  }



  std::vector<std::pair<types::global_dof_index, std::pair<double, double>>>
    vec_data;
};



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
class Poisson
{
private:
  void
  make_grid();
  void
  test_transfers(); // i'll leave it here for now, just to show what is wrong
                    // and what is correct
  void
  test_newidea_transfers();
  void
  assemble_system();
  void
  setup_multigrid();
  void
  solve();
  void
  output_results();


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

  std::pair<double, double>
  get_error() const;

  GridType        grid_type;
  PartitionerType partitioner_type;
  SolutionType    solution_type;
  unsigned int    extraction_level;
  double penalty_constant = 60.; // 10*(p+1)(p+d) for p = 1 and d = 2 => 60
  double l2_err;
  double semih1_err;

  DoFHandler<dim>                   finest_dof_handler;
  std::vector<SparseMatrix<double>> injection_matrices;
  std::vector<SparsityPattern>      injection_sparsity_patterns;
};


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



// Notes: when using the original grid some points finish in more than 1 cell.
// It should not be a problem but one should know that it happens. This is used
// to fill the transfer matrix using as the finest level the original
// triangulation
template <int dim>
void
fill_injection_transfer_matrix(
  const Mapping<dim>            &fine_mapping,
  const DoFHandler<dim>         &fine_dof_handler,
  const std::vector<Point<dim>> &coarse_support_points,
  const std::vector<Point<dim>> &fine_support_points,
  SparseMatrix<double>          &transfer_matrix,
  SparsityPattern               &sparsity_pattern)
{
  DynamicSparsityPattern dsp(fine_support_points.size(),
                             coarse_support_points.size());

  const FiniteElement<dim> &fe_fine = fine_dof_handler.get_fe();

  // Fill sparsity pattern
  for (unsigned int coarse_dof = 0; coarse_dof < coarse_support_points.size();
       ++coarse_dof)
    {
      // point in real space
      bool              found_cell    = false;
      const Point<dim> &support_point = coarse_support_points[coarse_dof];

      for (const auto &fine_cell : fine_dof_handler.active_cell_iterators())
        {
          if (fine_cell->point_inside(support_point))
            {
              // if (found_cell)
              //   cout << "Warning: support point " << support_point
              //        << " found in multiple fine cells." << std::endl;
              found_cell = true;
              std::vector<types::global_dof_index> fine_dof_indices(
                fe_fine.dofs_per_cell);
              fine_cell->get_dof_indices(fine_dof_indices);

              for (const auto fine_dof : fine_dof_indices)
                dsp.add(fine_dof, coarse_dof);

              break; // Found the cell, no need to continue, comment out
              // for debugging
              //  TODO: skip looping over all cells
            }
        }
      if (!found_cell)
        cout << "Warning: support point " << support_point
             << " not found in any fine cell." << std::endl;
    }

  sparsity_pattern.copy_from(dsp);
  transfer_matrix.reinit(sparsity_pattern);

  for (unsigned int coarse_dof = 0; coarse_dof < coarse_support_points.size();
       ++coarse_dof)
    {
      const Point<dim> &eval_point = coarse_support_points[coarse_dof];

      for (const auto &fine_cell : fine_dof_handler.active_cell_iterators())
        {
          if (fine_cell->point_inside(eval_point))
            {
              // Map physical point to reference coordinates
              Point<dim> ref_point =
                fine_mapping.transform_real_to_unit_cell(fine_cell, eval_point);

              std::vector<types::global_dof_index> fine_dof_indices(
                fe_fine.dofs_per_cell);
              fine_cell->get_dof_indices(fine_dof_indices);

              // Evaluate each shape function at the reference point
              for (unsigned int i = 0; i < fe_fine.dofs_per_cell; ++i)
                {
                  double shape_value = fe_fine.shape_value(i, ref_point);
                  transfer_matrix.set(fine_dof_indices[i],
                                      coarse_dof,
                                      shape_value);
                }

              break; // Found the cell, no need to continue
            }
        }
    }
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
  , penalty_constant(10. * (fe_degree + 1) * (fe_degree + dim))
  , finest_dof_handler(tria)
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
          tria.refine_global(2); // 4
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
      tria.refine_global(3);
#else
      Triangulation<dim> tria_hex;
      GridGenerator::hyper_cube(tria_hex, 0., 1.);
      tria_hex.refine_global(4);
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



// This function is still here just to show what happens if we don't do the post
// processing
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
        PolyUtils::constexpr_pow(2, dim); // 2^dim
      std::vector<Point<dim>> support_points_vector(dof_handler.n_dofs());

      DoFTools::map_dofs_to_support_points(mapping,
                                           dof_handler,
                                           support_points_vector);

      // auto start = std::chrono::system_clock::now();
      auto tree =
        pack_rtree<bgi::rstar<max_elem_per_node>>(support_points_vector);
      std::cout << "Total number of available levels: " << n_levels(tree)
                << std::endl;

#ifdef AGGLO_DEBUG
      // boost::geometry::index::detail::rtree::utilities::print(std::cout,
      // tree);
      Assert(n_levels(tree) >= 2,
             ExcMessage("At least two levels are needed."));
#endif
      // This part of the test is testing the interpolation and the finest
      // transfer
      bool                print_agglomerates = true;
      SolutionLinear<dim> support_function;
      Vector<double>      interpolated_sol_from_fine;
      {
        std::cout << "===================================================="
                  << std::endl;
        std::cout << "Testing between extraction level: " << extraction_level
                  << " and original triangulation" << std::endl;

        CellsAgglomerator<dim, decltype(tree), true> agglomerator{
          tree, extraction_level}; // This is used to test the interpolation on
                                   // extraction  level

        // vec_agglomerates is a vector of vectors of point in this case
        const auto vec_agglomerates = agglomerator.extract_agglomerates();
        std::cout << "Number of agglomerates to test interpolation: "
                  << vec_agglomerates.size() << std::endl;

        // Extracting finest coarse level
        CellsAgglomerator<dim, decltype(tree), true> coarse_agglomerator{
          tree,
          extraction_level}; // This is used to build the transfer matrix
                             // between the fine level and the extraction level
        const auto coarse_vec_agglomerates =
          coarse_agglomerator.extract_agglomerates();
        std::cout << "Number of agglomerates to test transfer: "
                  << coarse_vec_agglomerates.size() << std::endl;

        std::vector<BoundingBox<dim>> boxes;
        std::vector<BoundingBox<dim>> coarse_boxes;

        for (const auto &agglo : coarse_vec_agglomerates)
          coarse_boxes.emplace_back(agglo);

        for (const auto &agglo : vec_agglomerates)
          {
            boxes.emplace_back(agglo);

            if (print_agglomerates)
              {
                std::cout << "Point in agglomerate: \n";
                for (const auto &point : agglo)
                  {
                    std::cout << "p: " << point << "\t";
                  }
                std::cout << std::endl;
              }
          }

        // std::chrono::duration<double> wctduration =
        //   (std::chrono::system_clock::now() - start);
        // std::cout << "R-tree agglomerates built in " << wctduration.count()
        //           << " seconds [Wall Clock]" << std::endl;

        Vector<double> exact;
        // Checking if the interpolation is working on the extraction level
        {
          std::map<types::global_cell_index, types::global_cell_index>
            identity_mapping;
          for (unsigned int j = 0; j < boxes.size(); ++j)
            identity_mapping[j] = j;
          MappingBox<dim> mapping_box(boxes, identity_mapping);

          Triangulation<dim> dummy_tria;
          create_triangulation_from_bounding_boxes(dummy_tria, boxes);

          DoFHandler<dim> support_dof_handler(dummy_tria);
          FE_DGQ<dim>     support_dgfe(fe_q.get_degree());
          support_dof_handler.distribute_dofs(support_dgfe);

          Vector<double> support_vector(support_dof_handler.n_dofs());
          VectorTools::interpolate(mapping_box,
                                   support_dof_handler,
                                   support_function,
                                   support_vector);

          exact.reinit(support_dof_handler.n_dofs());
          exact = support_vector;

          // Output section
          DataOut<dim> data_out;
          data_out.attach_dof_handler(support_dof_handler);

          data_out.add_data_vector(support_vector, "interpolated_solution");

          Vector<float> cell_indices(dummy_tria.n_active_cells());
          for (const auto &cell : dummy_tria.active_cell_iterators())
            cell_indices[cell->active_cell_index()] = cell->active_cell_index();

          data_out.add_data_vector(cell_indices,
                                   "cell_index",
                                   DataOut<dim>::type_cell_data);

          data_out.build_patches(mapping_box, support_dgfe.get_degree() + 3);
          std::ofstream output("solution_comparison.vtu");
          data_out.write_vtu(output);
        }

        // Checking the transfer between extraction level and original grid

        std::map<types::global_cell_index, types::global_cell_index>
          coarse_identity_mapping;
        for (unsigned int j = 0; j < coarse_boxes.size(); ++j)
          coarse_identity_mapping[j] = j;
        MappingBox<dim> coarse_mapping_box(coarse_boxes,
                                           coarse_identity_mapping);

        Triangulation<dim> coarse_bbox_tria;
        create_triangulation_from_bounding_boxes(coarse_bbox_tria,
                                                 coarse_boxes);

        DoFHandler<dim> coarse_dof_handler(coarse_bbox_tria);
        FE_DGQ<dim>     coarse_dgfe(fe_q.get_degree());
        coarse_dof_handler.distribute_dofs(coarse_dgfe);

        std::vector<Point<dim>> coarse_support_points_vector(
          coarse_dof_handler.n_dofs());
        DoFTools::map_dofs_to_support_points(coarse_mapping_box,
                                             coarse_dof_handler,
                                             coarse_support_points_vector);
        SparsityPattern      transfer_sp;
        SparseMatrix<double> transfer_matrix;

        fill_injection_transfer_matrix(
          mapping, // Passo mapping perchè devo mappare le celle della griglia
                   // originale!
          dof_handler,
          coarse_support_points_vector,
          support_points_vector,
          transfer_matrix,
          transfer_sp);

        // transfer_matrix.print(std::cout);

        // Transfer matrix: fine (FE_Q) -> coarse (DG)
        std::cout << "Number of coarse support points: "
                  << coarse_support_points_vector.size() << std::endl;
        std::cout << "Number of fine support points: "
                  << support_points_vector.size() << std::endl;

        std::cout << "Transfer matrix size: " << transfer_matrix.m() << " x "
                  << transfer_matrix.n() << std::endl;

        // Sanity check ?
        Vector<double> support_vector(dof_handler.n_dofs());
        VectorTools::interpolate(mapping,
                                 dof_handler,
                                 support_function,
                                 support_vector);

        Vector<double> coarse_support_vector(coarse_dof_handler.n_dofs());
        transfer_matrix.Tvmult(coarse_support_vector, support_vector);

        interpolated_sol_from_fine.reinit(coarse_dof_handler.n_dofs());
        interpolated_sol_from_fine = coarse_support_vector;

        // Output section
        DataOut<dim> data_out;
        data_out.attach_dof_handler(coarse_dof_handler);

        data_out.add_data_vector(coarse_support_vector,
                                 "interpolated_solution");

        Vector<float> cell_indices(coarse_bbox_tria.n_active_cells());
        for (const auto &cell : coarse_bbox_tria.active_cell_iterators())
          cell_indices[cell->active_cell_index()] = cell->active_cell_index();

        data_out.add_data_vector(cell_indices,
                                 "cell_index",
                                 DataOut<dim>::type_cell_data);

        // Build patches and output
        data_out.build_patches(coarse_mapping_box,
                               coarse_dgfe.get_degree() + 3);
        std::ofstream output("solution_comparison_transfer.vtu");
        data_out.write_vtu(output);

        coarse_support_vector -= exact;
        std::cout
          << "L2 error between coarse interpolated solution and fine interpolated solution transferred to coarse: "
          << coarse_support_vector.l2_norm() << std::endl;
      }

      // Here i am testing transfers between internal levels
      {
        std::cout << "================================================"
                  << std::endl;
        std::cout << "Testing between internal level: " << n_levels(tree) - 1
                  << " and internal level: " << n_levels(tree) - 2 << std::endl;

        CellsAgglomerator<dim, decltype(tree), true> fine_agglomerator{
          tree, n_levels(tree) - 1};
        CellsAgglomerator<dim, decltype(tree), true> coarse_agglomerator{
          tree, n_levels(tree) - 2};
        const auto fine_vec_agglomerates =
          fine_agglomerator.extract_agglomerates();
        const auto coarse_vec_agglomerates =
          coarse_agglomerator.extract_agglomerates();
        std::cout << "N Agglomerates: " << fine_vec_agglomerates.size()
                  << ", N Coarse Agglomerates: "
                  << coarse_vec_agglomerates.size() << std::endl;

        std::vector<BoundingBox<dim>> fine_boxes;
        std::vector<BoundingBox<dim>> coarse_boxes;

        for (const auto &agglo : coarse_vec_agglomerates)
          coarse_boxes.emplace_back(agglo);

        for (const auto &agglo : fine_vec_agglomerates)
          fine_boxes.emplace_back(agglo);

        std::map<types::global_cell_index, types::global_cell_index>
          fine_identity_mapping;
        for (unsigned int j = 0; j < fine_boxes.size(); ++j)
          fine_identity_mapping[j] = j;

        std::map<types::global_cell_index, types::global_cell_index>
          coarse_identity_mapping;
        for (unsigned int j = 0; j < coarse_boxes.size(); ++j)
          coarse_identity_mapping[j] = j;

        MappingBox<dim> fine_mapping_box(fine_boxes, fine_identity_mapping);
        MappingBox<dim> coarse_mapping_box(coarse_boxes,
                                           coarse_identity_mapping);

        Triangulation<dim> fine_support_tria;
        Triangulation<dim> coarse_support_tria;
        create_triangulation_from_bounding_boxes(fine_support_tria, fine_boxes);
        create_triangulation_from_bounding_boxes(coarse_support_tria,
                                                 coarse_boxes);

        DoFHandler<dim> fine_dof_handler(fine_support_tria);
        DoFHandler<dim> coarse_dof_handler(coarse_support_tria);

        FE_DGQ<dim> support_dgfe(fe_q.get_degree());

        fine_dof_handler.distribute_dofs(support_dgfe);
        coarse_dof_handler.distribute_dofs(support_dgfe);

        std::vector<Point<dim>> fine_support_points_vector(
          fine_dof_handler.n_dofs());
        std::vector<Point<dim>> coarse_support_points_vector(
          coarse_dof_handler.n_dofs());

        DoFTools::map_dofs_to_support_points(fine_mapping_box,
                                             fine_dof_handler,
                                             fine_support_points_vector);
        DoFTools::map_dofs_to_support_points(coarse_mapping_box,
                                             coarse_dof_handler,
                                             coarse_support_points_vector);

        SparsityPattern      transfer_sp;
        SparseMatrix<double> transfer_matrix;

        // Transfer matrix: fine (DG_Q) -> coarse (DG_Q)
        fill_injection_transfer_matrix(fine_mapping_box,
                                       fine_dof_handler,
                                       coarse_support_points_vector,
                                       fine_support_points_vector,
                                       transfer_matrix,
                                       transfer_sp);

        // transfer_matrix.print(std::cout);

        std::cout << "Number of coarse support points: "
                  << coarse_support_points_vector.size() << std::endl;
        std::cout << "Number of fine support points: "
                  << fine_support_points_vector.size() << std::endl;

        std::cout << "Transfer matrix size: " << transfer_matrix.m() << " x "
                  << transfer_matrix.n() << std::endl;

        Vector<double> fine_interpolated_sol(fine_dof_handler.n_dofs());
        Vector<double> coarse_transferred_sol(coarse_dof_handler.n_dofs());

        VectorTools::interpolate(fine_mapping_box,
                                 fine_dof_handler,
                                 support_function,
                                 fine_interpolated_sol);

        transfer_matrix.Tvmult(coarse_transferred_sol, fine_interpolated_sol);

        // Output fine interpolated solution
        {
          DataOut<dim> data_out;
          data_out.attach_dof_handler(fine_dof_handler);

          data_out.add_data_vector(fine_interpolated_sol,
                                   "interpolated_solution");

          Vector<float> cell_indices(fine_support_tria.n_active_cells());
          for (const auto &cell : fine_support_tria.active_cell_iterators())
            cell_indices[cell->active_cell_index()] = cell->active_cell_index();

          data_out.add_data_vector(cell_indices,
                                   "cell_index",
                                   DataOut<dim>::type_cell_data);

          data_out.build_patches(fine_mapping_box,
                                 support_dgfe.get_degree() + 3);
          std::ofstream output("fine_level_interpolant.vtu");
          data_out.write_vtu(output);
        }
        // Output coarse transferred solution
        {
          DataOut<dim> data_out;
          data_out.attach_dof_handler(coarse_dof_handler);

          data_out.add_data_vector(coarse_transferred_sol,
                                   "interpolated_solution");

          Vector<float> cell_indices(coarse_support_tria.n_active_cells());
          for (const auto &cell : coarse_support_tria.active_cell_iterators())
            cell_indices[cell->active_cell_index()] = cell->active_cell_index();

          data_out.add_data_vector(cell_indices,
                                   "cell_index",
                                   DataOut<dim>::type_cell_data);

          data_out.build_patches(coarse_mapping_box,
                                 support_dgfe.get_degree() + 3);
          std::ofstream output("coarse_level_interpolant_transferred.vtu");
          data_out.write_vtu(output);
        }

        coarse_transferred_sol -= interpolated_sol_from_fine;
        std::cout
          << "L2 error between interpolated solution from internal extraction levels and interpolated solution on original tria transferred to coarse: "
          << coarse_transferred_sol.l2_norm() << std::endl;
      }

      // Check number of agglomerates
      // if constexpr (dim == 2)
      //   {
      //     GridOut           grid_out_svg;
      //     GridOutFlags::Svg svg_flags;
      //     svg_flags.background     =
      //     GridOutFlags::Svg::Background::transparent;
      //     svg_flags.line_thickness = 1;
      //     svg_flags.boundary_line_thickness = 1;
      //     svg_flags.label_subdomain_id      = true;
      //     svg_flags.coloring =
      //       GridOutFlags::Svg::subdomain_id; // GridOutFlags::Svg::none
      //     grid_out_svg.set_flags(svg_flags);
      //     std::string   grid_type = "agglomerated_grid";
      //     std::ofstream out(grid_type + ".svg");
      //     grid_out_svg.write_svg(tria, out);
      //   }
    }
}



template <int dim>
void
Poisson<dim>::test_newidea_transfers()
{
  if (partitioner_type == PartitionerType::rtree)
    {
      DoFHandler<dim> dof_handler(tria); // This is the finest DoF_Handler
      dof_handler.distribute_dofs(fe_q);

      namespace bgi = boost::geometry::index;
      static constexpr unsigned int max_elem_per_node =
        PolyUtils::constexpr_pow(2, dim); // 2^dim
      std::vector<Point<dim>> support_points_vector(dof_handler.n_dofs());

      DoFTools::map_dofs_to_support_points(mapping,
                                           dof_handler,
                                           support_points_vector);

      // auto start = std::chrono::system_clock::now();
      auto tree =
        pack_rtree<bgi::rstar<max_elem_per_node>>(support_points_vector);
      std::cout
        << "======================= Testing new ideas ==================="
        << std::endl;
      std::cout << "Total number of available levels: " << n_levels(tree)
                << std::endl;
      std::vector<std::vector<BoundingBox<dim>>> all_level_boxes(
        n_levels(tree)); // N. B. there is an off by 1. all_level_boxes[0] =
                         // boxes at level 1  of the tree

      for (unsigned int i = 0; i < n_levels(tree); ++i)
        {
          CellsAgglomerator<dim, decltype(tree), true> agglomerator{tree,
                                                                    i + 1};
          const auto agglomerates = agglomerator.extract_agglomerates();
          all_level_boxes[i].reserve(agglomerates.size());
          for (const auto &agglo : agglomerates)
            all_level_boxes[i].emplace_back(agglo);
          std::cout << "Level " << i + 1
                    << " number of agglomerates: " << all_level_boxes[i].size()
                    << std::endl;
        }

      std::cout
        << "------------------------Degen boxes handling----------------------------"
        << std::endl;

      for (unsigned int i = 0; i < n_levels(tree); ++i)
        {
          std::cout << "Checking for degenerate Bounding Boxes at level "
                    << i + 1 << std::endl;
          unsigned int              degenerate_count = 0;
          std::vector<unsigned int> degenerate_indices;

          for (unsigned int bbox_idx = 0; bbox_idx < all_level_boxes[i].size();
               ++bbox_idx)
            {
              const auto &bbox = all_level_boxes[i][bbox_idx];
              if (bbox.volume() < 1e-12)
                {
                  // std::cout << "Warning: degenerate bbox with volume "
                  //           << bbox.volume() << " at level " << i + 1
                  //           << ", bbox idx " << bbox_idx << std::endl;
                  degenerate_count++;
                  degenerate_indices.push_back(bbox_idx);
                }
            }

          if (degenerate_indices.empty())
            std::cout << "No degenerate bounding boxes found at level " << i + 1
                      << std::endl;
          else
            std::cout << "Total number of degenerate bounding boxes at level "
                      << i + 1 << ": " << degenerate_count << std::endl;

          for (auto degen_bbox_idx : degenerate_indices)
            {
              double       min_distance = std::numeric_limits<double>::max();
              unsigned int closest_bbox_idx = degen_bbox_idx;
              for (unsigned int bbox_idx = 0;
                   bbox_idx < all_level_boxes[i].size();
                   ++bbox_idx)
                {
                  // Not efficient but whatever, if the bbox is degenerate skip
                  // it
                  if (std::find(degenerate_indices.begin(),
                                degenerate_indices.end(),
                                bbox_idx) != degenerate_indices.end())
                    continue;
                  else
                    {
                      Point<dim> degen_center =
                        all_level_boxes[i][degen_bbox_idx].center();
                      auto bbox_dist =
                        all_level_boxes[i][bbox_idx].signed_distance(
                          degen_center);
                      if (bbox_dist < min_distance)
                        {
                          min_distance     = bbox_dist;
                          closest_bbox_idx = bbox_idx;
                        }
                    }
                }
              if (closest_bbox_idx != degen_bbox_idx)
                {
                  std::cout << "Merging degenerate bbox idx " << degen_bbox_idx
                            << " into closest bbox idx " << closest_bbox_idx
                            << std::endl;
                  all_level_boxes[i][closest_bbox_idx].merge_with(
                    all_level_boxes[i][degen_bbox_idx]);
                }
              else
                {
                  std::cout
                    << "Warning: degenerate bbox has no other non-degenerate bbox to merge with."
                    << std::endl;
                }
            }
          std::vector<BoundingBox<dim>> new_level_boxes;
          new_level_boxes.reserve(all_level_boxes[i].size());
          for (unsigned int bbox_idx = 0; bbox_idx < all_level_boxes[i].size();
               ++bbox_idx)
            {
              if (std::find(degenerate_indices.begin(),
                            degenerate_indices.end(),
                            bbox_idx) == degenerate_indices.end())
                {
                  new_level_boxes.push_back(all_level_boxes[i][bbox_idx]);
                }
            }
          all_level_boxes[i] = std::move(new_level_boxes);

          std::cout << "Checking again for degenerate Bounding Boxes at level "
                    << i + 1 << std::endl;
          bool found_wrong_one = false;
          for (const auto &bbox : all_level_boxes[i])
            {
              if (bbox.volume() < 1e-12)
                {
                  std::cout << "Error: degenerate bbox still present after "
                               "merging at level "
                            << i + 1 << std::endl;
                  found_wrong_one = true;
                }
            }
          if (!found_wrong_one)
            std::cout << "No degenerate bounding boxes found at level " << i + 1
                      << " after merging." << std::endl;
        }

      std::cout
        << "-----------------------Support points outside boxes fix-----------------------------"
        << std::endl;

      for (unsigned int i = 0; i < n_levels(tree) - 1; ++i)
        {
          std::cout << "Checking support points at level " << i + 1
                    << " inside agglomerates at level " << i + 2 << std::endl;
          std::map<types::global_cell_index, types::global_cell_index>
            coarse_identity_mapping;
          for (unsigned int j = 0; j < all_level_boxes[i].size(); ++j)
            coarse_identity_mapping[j] = j;

          MappingBox<dim>    coarse_mapping_box(all_level_boxes[i],
                                             coarse_identity_mapping);
          Triangulation<dim> coarse_tria;
          create_triangulation_from_bounding_boxes(coarse_tria,
                                                   all_level_boxes[i]);
          DoFHandler<dim> coarse_dof_handler(coarse_tria);
          FE_DGQ<dim>     coarse_dgfe(fe_q.get_degree());
          coarse_dof_handler.distribute_dofs(coarse_dgfe);
          std::vector<Point<dim>> coarse_support_points_vector(
            coarse_dof_handler.n_dofs());
          DoFTools::map_dofs_to_support_points(coarse_mapping_box,
                                               coarse_dof_handler,
                                               coarse_support_points_vector);

          // This maps which coarse support points are outside the next
          // level agglomerates and the closes bbox_idx to it
          std::map<unsigned int, unsigned int> closest_bbox;

          for (unsigned int j = 0; j < coarse_support_points_vector.size(); ++j)
            {
              double       distance = std::numeric_limits<double>::max();
              unsigned int idx_closest_bbox;
              for (unsigned int idx = 0; idx < all_level_boxes[i + 1].size();
                   ++idx)
                {
                  const auto &bbox = all_level_boxes[i + 1][idx];

                  if (distance >
                      bbox.signed_distance(coarse_support_points_vector[j]))
                    {
                      distance =
                        bbox.signed_distance(coarse_support_points_vector[j]);
                      idx_closest_bbox = idx;
                    }
                }
              if (distance > 0.)
                {
                  // std::cout
                  //   << "Warning: coarse support point "
                  //   << coarse_support_points_vector[j] << " at level "
                  //   <<
                  //   i
                  //   +
                  //   1
                  //   << " is outside the agglomerates of the next level
                  //   by distance "
                  //   << distance << std::endl;
                  closest_bbox[j] = idx_closest_bbox;
                }
            }
          std::cout
            << "Number of coarse support points outside next level agglomerates: "
            << closest_bbox.size() << std::endl;

          for (const auto &pair : closest_bbox)
            {
              const unsigned int coarse_support_point_idx = pair.first;
              const unsigned int bbox_idx                 = pair.second;

              BoundingBox<dim> degen_bbox(
                coarse_support_points_vector[coarse_support_point_idx]);
              // std::cout << "new bbox vertexes "
              //           << degen_bbox.get_boundary_points().first << "
              //           -
              //           "
              //           << degen_bbox.get_boundary_points().second <<
              //           std::endl;
              // std::cout
              //   << "merging into bbox idx " << bbox_idx << " with
              //   vertexes
              //   "
              //   << all_level_boxes[i +
              //   1][bbox_idx].get_boundary_points().first
              //   << " - "
              //   << all_level_boxes[i +
              //   1][bbox_idx].get_boundary_points().second
              //   << std::endl;
              all_level_boxes[i + 1][bbox_idx].merge_with(degen_bbox);
              // std::cout
              //   << "merged bbox vertexes "
              //   << all_level_boxes[i +
              //   1][bbox_idx].get_boundary_points().first
              //   << " - "
              //   << all_level_boxes[i +
              //   1][bbox_idx].get_boundary_points().second
              //   << std::endl;
            }
        }

      std::cout
        << "----------------------Interpolation between levels check-------------------------"
        << std::endl;

      std::vector<BoundingBox<dim>> fine_boxes =
        all_level_boxes[all_level_boxes.size() - 1];
      std::vector<BoundingBox<dim>> coarse_boxes =
        all_level_boxes[all_level_boxes.size() - 2];

      std::map<types::global_cell_index, types::global_cell_index>
        fine_identity_mapping;
      for (unsigned int j = 0; j < fine_boxes.size(); ++j)
        fine_identity_mapping[j] = j;

      std::map<types::global_cell_index, types::global_cell_index>
        coarse_identity_mapping;
      for (unsigned int j = 0; j < coarse_boxes.size(); ++j)
        coarse_identity_mapping[j] = j;

      MappingBox<dim> fine_mapping_box(fine_boxes, fine_identity_mapping);
      MappingBox<dim> coarse_mapping_box(coarse_boxes, coarse_identity_mapping);

      Triangulation<dim> fine_support_tria;
      Triangulation<dim> coarse_support_tria;
      create_triangulation_from_bounding_boxes(fine_support_tria, fine_boxes);
      create_triangulation_from_bounding_boxes(coarse_support_tria,
                                               coarse_boxes);

      DoFHandler<dim> fine_dof_handler(fine_support_tria);
      DoFHandler<dim> coarse_dof_handler(coarse_support_tria);

      FE_DGQ<dim> support_dgfe(fe_q.get_degree());

      fine_dof_handler.distribute_dofs(support_dgfe);
      coarse_dof_handler.distribute_dofs(support_dgfe);

      std::vector<Point<dim>> fine_support_points_vector(
        fine_dof_handler.n_dofs());
      std::vector<Point<dim>> coarse_support_points_vector(
        coarse_dof_handler.n_dofs());

      DoFTools::map_dofs_to_support_points(fine_mapping_box,
                                           fine_dof_handler,
                                           fine_support_points_vector);
      DoFTools::map_dofs_to_support_points(coarse_mapping_box,
                                           coarse_dof_handler,
                                           coarse_support_points_vector);

      SparsityPattern      transfer_sp;
      SparseMatrix<double> transfer_matrix;

      // Transfer matrix: fine (DG_Q) -> coarse (DG_Q)
      fill_injection_transfer_matrix(fine_mapping_box,
                                     fine_dof_handler,
                                     coarse_support_points_vector,
                                     fine_support_points_vector,
                                     transfer_matrix,
                                     transfer_sp);

      std::cout << "Number of coarse support points: "
                << coarse_support_points_vector.size() << std::endl;
      std::cout << "Number of fine support points: "
                << fine_support_points_vector.size() << std::endl;

      std::cout << "Transfer matrix size: " << transfer_matrix.m() << " x "
                << transfer_matrix.n() << std::endl;

      SolutionLinear<dim> support_function;

      Vector<double> fine_interpolated_sol(fine_dof_handler.n_dofs());
      Vector<double> coarse_transferred_sol(coarse_dof_handler.n_dofs());

      VectorTools::interpolate(fine_mapping_box,
                               fine_dof_handler,
                               support_function,
                               fine_interpolated_sol);

      transfer_matrix.Tvmult(coarse_transferred_sol, fine_interpolated_sol);

      // Output fine interpolated solution
      {
        DataOut<dim> data_out;
        data_out.attach_dof_handler(fine_dof_handler);

        data_out.add_data_vector(fine_interpolated_sol,
                                 "interpolated_solution");

        Vector<float> cell_indices(fine_support_tria.n_active_cells());
        for (const auto &cell : fine_support_tria.active_cell_iterators())
          cell_indices[cell->active_cell_index()] = cell->active_cell_index();

        data_out.add_data_vector(cell_indices,
                                 "cell_index",
                                 DataOut<dim>::type_cell_data);

        data_out.build_patches(fine_mapping_box, support_dgfe.get_degree() + 3);
        std::ofstream output("new_strat_fine_level_interpolant.vtu");
        data_out.write_vtu(output);
      }
      // Output coarse transferred solution
      {
        DataOut<dim> data_out;
        data_out.attach_dof_handler(coarse_dof_handler);

        data_out.add_data_vector(coarse_transferred_sol,
                                 "interpolated_solution");

        Vector<float> cell_indices(coarse_support_tria.n_active_cells());
        for (const auto &cell : coarse_support_tria.active_cell_iterators())
          cell_indices[cell->active_cell_index()] = cell->active_cell_index();

        data_out.add_data_vector(cell_indices,
                                 "cell_index",
                                 DataOut<dim>::type_cell_data);

        data_out.build_patches(coarse_mapping_box,
                               support_dgfe.get_degree() + 3);
        std::ofstream output(
          "new_strat_coarse_level_interpolant_transferred.vtu");
        data_out.write_vtu(output);
      }

      // Let's check against the solution interpolated from the original
      // tria
      Vector<double> interpolated_sol_from_fine;
      {
        DoFHandler<dim> original_dof_handler(tria);
        original_dof_handler.distribute_dofs(fe_q);
        Vector<double> original_interpolated_sol(original_dof_handler.n_dofs());
        VectorTools::interpolate(mapping,
                                 original_dof_handler,
                                 support_function,
                                 original_interpolated_sol);

        interpolated_sol_from_fine.reinit(coarse_dof_handler.n_dofs());

        SparsityPattern      transfer_sp;
        SparseMatrix<double> transfer_matrix;
        fill_injection_transfer_matrix(mapping,
                                       original_dof_handler,
                                       coarse_support_points_vector,
                                       support_points_vector,
                                       transfer_matrix,
                                       transfer_sp);
        transfer_matrix.Tvmult(interpolated_sol_from_fine,
                               original_interpolated_sol);
      }
      coarse_transferred_sol -= interpolated_sol_from_fine;
      std::cout
        << "L2 error between interpolated solution from internal extraction levels and interpolated solution on original tria transferred to coarse: "
        << coarse_transferred_sol.l2_norm() << std::endl;
    }
}



template <int dim>
void
Poisson<dim>::assemble_system()
{
  finest_dof_handler.distribute_dofs(fe_q);

  // assembling standard Poisson system
  dsp.reinit(finest_dof_handler.n_dofs(), finest_dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(finest_dof_handler, dsp);
  sparsity.copy_from(dsp);
  system_matrix.reinit(sparsity);

  solution.reinit(finest_dof_handler.n_dofs());
  system_rhs.reinit(finest_dof_handler.n_dofs());

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

  for (const auto &cell : finest_dof_handler.active_cell_iterators())
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
  VectorTools::interpolate_boundary_values(finest_dof_handler,
                                           types::boundary_id(0),
                                           *analytical_solution,
                                           boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution,
                                     system_rhs);

  std::cout << "Built finest system matrix" << std::endl;
}



// WIP
template <int dim>
void
Poisson<dim>::setup_multigrid()
{
  // Setup rtree with support points and modify the bboxes
  namespace bgi = boost::geometry::index;
  static constexpr unsigned int max_elem_per_node =
    PolyUtils::constexpr_pow(2, dim); // 2^dim
  std::vector<Point<dim>> support_points_vector(finest_dof_handler.n_dofs());

  DoFTools::map_dofs_to_support_points(mapping,
                                       finest_dof_handler,
                                       support_points_vector);

  auto tree = pack_rtree<bgi::rstar<max_elem_per_node>>(support_points_vector);
  std::cout << "======================= Multigrid testing ==================="
            << std::endl;
  std::vector<std::vector<BoundingBox<dim>>> all_level_boxes(
    n_levels(tree)); // N. B. there is an off by 1. all_level_boxes[0] =
                     // boxes at level 1  of the tree

  // This cycle creates all the bounding boxes at each level
  for (unsigned int i = 0; i < n_levels(tree); ++i)
    {
      CellsAgglomerator<dim, decltype(tree), true> agglomerator{tree, i + 1};
      const auto agglomerates = agglomerator.extract_agglomerates();
      all_level_boxes[i].reserve(agglomerates.size());
      for (const auto &agglo : agglomerates)
        all_level_boxes[i].emplace_back(agglo);
    } // Bbox creator

  // This cycle deals with degenerate boxes at each level by merging them into
  // the closest bbox
  for (unsigned int i = 0; i < n_levels(tree); ++i)
    {
      std::vector<unsigned int> degenerate_indices;

      for (unsigned int bbox_idx = 0; bbox_idx < all_level_boxes[i].size();
           ++bbox_idx)
        {
          const auto &bbox = all_level_boxes[i][bbox_idx];
          if (bbox.volume() < 1e-12)
            degenerate_indices.push_back(bbox_idx);
        }

      for (auto degen_bbox_idx : degenerate_indices)
        {
          double       min_distance     = std::numeric_limits<double>::max();
          unsigned int closest_bbox_idx = degen_bbox_idx;
          for (unsigned int bbox_idx = 0; bbox_idx < all_level_boxes[i].size();
               ++bbox_idx)
            {
              // Not efficient but whatever, if the bbox is degenerate skip
              // it
              if (std::find(degenerate_indices.begin(),
                            degenerate_indices.end(),
                            bbox_idx) != degenerate_indices.end())
                continue;
              else
                {
                  Point<dim> degen_center =
                    all_level_boxes[i][degen_bbox_idx].center();
                  auto bbox_dist =
                    all_level_boxes[i][bbox_idx].signed_distance(degen_center);
                  if (bbox_dist < min_distance)
                    {
                      min_distance     = bbox_dist;
                      closest_bbox_idx = bbox_idx;
                    }
                }
            }
          if (closest_bbox_idx != degen_bbox_idx)
            all_level_boxes[i][closest_bbox_idx].merge_with(
              all_level_boxes[i][degen_bbox_idx]);

          else
            {
              std::cout
                << "Warning: degenerate bbox has no other non-degenerate bbox to merge with."
                << std::endl;
            }
        }
      std::vector<BoundingBox<dim>> new_level_boxes;
      new_level_boxes.reserve(all_level_boxes[i].size());
      for (unsigned int bbox_idx = 0; bbox_idx < all_level_boxes[i].size();
           ++bbox_idx)
        {
          if (std::find(degenerate_indices.begin(),
                        degenerate_indices.end(),
                        bbox_idx) == degenerate_indices.end())
            {
              new_level_boxes.push_back(all_level_boxes[i][bbox_idx]);
            }
        }
      all_level_boxes[i] = std::move(new_level_boxes);

      for (unsigned int bbox_idx = 0; bbox_idx < all_level_boxes[i].size();
           ++bbox_idx)
        {
          const auto &bbox = all_level_boxes[i][bbox_idx];
          if (bbox.volume() < 1e-12)
            std::cout
              << "Error: degenerate bbox still present after merging at level "
              << i + 1 << std::endl;
        }
    } // Degenerate box handler

  std::vector<std::unique_ptr<Triangulation<dim>>> all_level_triangulations;
  all_level_triangulations.reserve(
    n_levels(tree)); // Needed to keep alive the trias for the DoFHandlers

  std::vector<std::unique_ptr<DoFHandler<dim>>> all_level_support_DoFHandlers;
  all_level_support_DoFHandlers.reserve(n_levels(tree));
  std::vector<MappingBox<dim>> all_level_mapping_boxes;
  all_level_mapping_boxes.reserve(n_levels(tree));
  std::vector<std::vector<Point<dim>>> all_level_support_points_vectors(
    n_levels(tree));

  // This cycle deals with support points outside the next level boxes by
  // extending the closest box to a support point in order to include it
  // It also starts filling the DoFHandlers, mapping boxes and support points
  // for each level needed for the transfer operators
  for (unsigned int i = 0; i < n_levels(tree) - 1; ++i)
    {
      std::map<types::global_cell_index, types::global_cell_index>
        coarse_identity_mapping;
      for (unsigned int j = 0; j < all_level_boxes[i].size(); ++j)
        coarse_identity_mapping[j] = j;

      all_level_mapping_boxes.emplace_back(all_level_boxes[i],
                                           coarse_identity_mapping);

      all_level_triangulations.push_back(
        std::make_unique<Triangulation<dim>>());
      create_triangulation_from_bounding_boxes(*all_level_triangulations[i],
                                               all_level_boxes[i]);
      all_level_support_DoFHandlers.push_back(
        std::make_unique<DoFHandler<dim>>(*all_level_triangulations[i]));
      FE_DGQ<dim> coarse_dgfe(fe_q.get_degree());
      all_level_support_DoFHandlers[i]->distribute_dofs(coarse_dgfe);
      std::vector<Point<dim>> coarse_support_points_vector(
        all_level_support_DoFHandlers[i]->n_dofs());

      DoFTools::map_dofs_to_support_points(all_level_mapping_boxes[i],
                                           *all_level_support_DoFHandlers[i],
                                           coarse_support_points_vector);
      all_level_support_points_vectors[i] = coarse_support_points_vector;

      // This maps which coarse support points are outside the next
      // level agglomerates and the closes bbox_idx to it
      std::map<unsigned int, unsigned int> closest_bbox;

      for (unsigned int j = 0; j < coarse_support_points_vector.size(); ++j)
        {
          double       distance = std::numeric_limits<double>::max();
          unsigned int idx_closest_bbox;
          for (unsigned int idx = 0; idx < all_level_boxes[i + 1].size(); ++idx)
            {
              const auto &bbox = all_level_boxes[i + 1][idx];

              if (distance >
                  bbox.signed_distance(coarse_support_points_vector[j]))
                {
                  distance =
                    bbox.signed_distance(coarse_support_points_vector[j]);
                  idx_closest_bbox = idx;
                }
            }
          if (distance > 0.)
            {
              closest_bbox[j] = idx_closest_bbox;
            }
        }

      for (const auto &pair : closest_bbox)
        {
          const unsigned int coarse_support_point_idx = pair.first;
          const unsigned int bbox_idx                 = pair.second;

          BoundingBox<dim> degen_bbox(
            coarse_support_points_vector[coarse_support_point_idx]);

          all_level_boxes[i + 1][bbox_idx].merge_with(degen_bbox);
        }
    } // Support points outside boxes handler

  // Filling the data for the finest level
  {
    std::map<types::global_cell_index, types::global_cell_index>
      finest_identity_mapping;
    for (unsigned int j = 0; j < all_level_boxes[n_levels(tree) - 1].size();
         ++j)
      finest_identity_mapping[j] = j;

    all_level_mapping_boxes.emplace_back(all_level_boxes[n_levels(tree) - 1],
                                         finest_identity_mapping);

    all_level_triangulations.push_back(std::make_unique<Triangulation<dim>>());
    create_triangulation_from_bounding_boxes(
      *all_level_triangulations[n_levels(tree) - 1],
      all_level_boxes[n_levels(tree) - 1]);
    all_level_support_DoFHandlers.push_back(std::make_unique<DoFHandler<dim>>(
      *all_level_triangulations[n_levels(tree) - 1]));
    FE_DGQ<dim> finest_dgfe(fe_q.get_degree());
    all_level_support_DoFHandlers[n_levels(tree) - 1]->distribute_dofs(
      finest_dgfe);
    std::vector<Point<dim>> finest_support_points_vector(
      all_level_support_DoFHandlers[n_levels(tree) - 1]->n_dofs());

    DoFTools::map_dofs_to_support_points(
      all_level_mapping_boxes[n_levels(tree) - 1],
      *all_level_support_DoFHandlers[n_levels(tree) - 1],
      finest_support_points_vector);
    all_level_support_points_vectors[n_levels(tree) - 1] =
      finest_support_points_vector;
  } // finest level data filler

  // Check that sizes of the data structures are consistent
  AssertThrow(all_level_support_DoFHandlers.size() == n_levels(tree),
              ExcMessage(
                "Inconsistent number of DoFHandlers for multigrid levels"));
  AssertThrow(all_level_mapping_boxes.size() == n_levels(tree),
              ExcMessage("Inconsistent number of MappingBoxes for multigrid "
                         "levels"));
  AssertThrow(all_level_support_points_vectors.size() == n_levels(tree),
              ExcMessage("Inconsistent number of support points vectors for "
                         "multigrid levels"));

  injection_matrices.resize(n_levels(tree));
  injection_sparsity_patterns.resize(n_levels(tree));

  for (unsigned int level = 0; level < n_levels(tree) - 1; ++level)
    {
      fill_injection_transfer_matrix(
        all_level_mapping_boxes[level + 1],
        *all_level_support_DoFHandlers[level + 1],
        all_level_support_points_vectors[level],
        all_level_support_points_vectors[level + 1],
        injection_matrices[level],
        injection_sparsity_patterns[level]);
    }
  // Fill the transfer between original grid and finest level
  {
    fill_injection_transfer_matrix(
      mapping,
      finest_dof_handler,
      all_level_support_points_vectors[n_levels(tree) - 1],
      support_points_vector,
      injection_matrices[n_levels(tree) - 1],
      injection_sparsity_patterns[n_levels(tree) - 1]);
  }

  std::cout << "Finished setting up multigrid transfer operators" << std::endl;

  unsigned int level_counter = 0;
  for (const auto &sup_points : all_level_support_points_vectors)
    {
      std::cout << "Support points vector size: " << sup_points.size()
                << " at level " << level_counter++ + 1 << std::endl;
    }
  std::cout << "Support points vector size original tria: "
            << support_points_vector.size() << std::endl;

  for (const auto &mat : injection_matrices)
    {
      std::cout << "Injection matrix size: " << mat.m() << " x " << mat.n()
                << std::endl;
    }
}


template <int dim>
void
Poisson<dim>::solve()
{
  SparseDirectUMFPACK A_direct;
  A_direct.initialize(system_matrix);
  A_direct.vmult(solution, system_rhs);
}



template <int dim>
void
Poisson<dim>::output_results()
{}



template <int dim>
inline std::pair<double, double>
Poisson<dim>::get_error() const
{
  return std::make_pair(l2_err, semih1_err);
}



template <int dim>
void
Poisson<dim>::run()
{
  make_grid();
  // test_transfers();
  test_newidea_transfers();
  auto start = std::chrono::high_resolution_clock::now();
  assemble_system();
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
    std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  std::cout << "Time taken by assemble_system(): " << duration.count() / 1e6
            << " seconds" << std::endl;

  setup_multigrid();
  // solve();
  output_results();
}



int
main()
{
  // Testing p-convergence
  // ConvergenceInfo convergence_info;
  // std::cout << "Testing p-convergence" << std::endl;
  {
#ifdef HEX
    for (unsigned int fe_degree : {1})
#else
    for (unsigned int fe_degree : {1, 2, 3})
#endif
      {
        std::cout << "Fe degree: " << fe_degree << std::endl;
        Poisson<2> poisson_problem{GridType::unstructured,
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
