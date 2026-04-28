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
#include <deal.II/base/index_set.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <agglomeration_handler.h>
#include <fe_agglodgp.h>
#include <poly_utils.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

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

namespace
{
  std::string
  sanitize_for_filename(const std::string &input)
  {
    std::string sanitized;
    sanitized.reserve(input.size());
    for (const unsigned char c : input)
      {
        if (std::isalnum(c) || c == '-' || c == '_')
          sanitized.push_back(static_cast<char>(c));
        else
          sanitized.push_back('_');
      }
    return sanitized.empty() ? "run" : sanitized;
  }


  std::string
  current_time_string()
  {
    const auto        now = std::chrono::system_clock::now();
    const std::time_t tt  = std::chrono::system_clock::to_time_t(now);
    std::tm           tm_buf{};
#if defined(_WIN32)
    localtime_s(&tm_buf, &tt);
#else
    localtime_r(&tt, &tm_buf);
#endif
    std::ostringstream out;
    out << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
    return out.str();
  }


  void
  write_stat_line(std::ofstream     &file,
                  const std::string &label,
                  const std::string &value)
  {
    file << "  " << std::left << std::setw(46) << label << " : " << value
         << std::endl;
  }
} // namespace



// enum SolutionType
// {
//   linear,      // x+y-1
//   quadratic,   // x^2+y^2-1
//   product,     // xy(x-1)(y-1)
//   product_sine // sin(pi*x)*sin(pi*y)
// };



template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide(const std::string &sol_type = "linear")
    : Function<dim>()
  {
    solution_type = sol_type;
  }

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double>           &values,
             const unsigned int /*component*/) const override;

private:
  std::string solution_type;
};



template <int dim>
void
RightHandSide<dim>::value_list(const std::vector<Point<dim>> &points,
                               std::vector<double>           &values,
                               const unsigned int /*component*/) const
{
  if (solution_type == "linear")
    {
      for (unsigned int i = 0; i < values.size(); ++i)
        values[i] = 0.; // Laplacian of linear function
    }
  else if (solution_type == "quadratic")
    {
      for (unsigned int i = 0; i < values.size(); ++i)
        values[i] = -2.0 * dim; // -Δ(Σ x_d^2 - 1) = -2*dim
    }
  else if (solution_type == "product")
    {
      for (unsigned int i = 0; i < values.size(); ++i)
        values[i] = -2. * points[i][0] * (points[i][0] - 1.) -
                    2. * points[i][1] * (points[i][1] - 1.);
    }
  else if (solution_type == "product_sine")
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
SolutionQuadratic<dim>::value(const Point<dim> &p, const unsigned int) const
{
  double s = 0.;
  for (unsigned int d = 0; d < dim; ++d)
    s += p[d] * p[d];
  return s - 1.;
}

template <int dim>
Tensor<1, dim>
SolutionQuadratic<dim>::gradient(const Point<dim> &p, const unsigned int) const
{
  Tensor<1, dim> return_value;
  for (unsigned int d = 0; d < dim; ++d)
    return_value[d] = 2. * p[d];
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
class ProblemParameters : public ParameterAcceptor
{
public:
  ProblemParameters();

  std::string  output_directory    = ".";
  unsigned int extraction_level    = 1;
  unsigned int mg_starting_level   = 2;
  unsigned int smoother_steps      = 1;
  bool         use_piston          = false;
  unsigned int fe_degree           = 1;
  unsigned int coarse_fe_degree    = 1;
  std::string  grid_type           = "unstructured";
  std::string  partitioner_type    = "rtree";
  std::string  solution_type       = "linear";
  unsigned int n_refinements       = 1;
  unsigned int n_ref_cycles        = 1;
  bool         keep_ratio_constant = false; // try to keep H/h fixed

  mutable ParameterAcceptorProxy<ReductionControl> outer_solver_control;
};

template <int dim>
ProblemParameters<dim>::ProblemParameters()
  : ParameterAcceptor("R-tree based MG/")
  , outer_solver_control("Reduction control")

{
  add_parameter("Finite element degree", fe_degree);
  add_parameter("Output directory", output_directory);
  add_parameter("Solution type", solution_type);
  add_parameter("Coarse Finite element degree", coarse_fe_degree);

  enter_subsection("Grid generation");
  {
    add_parameter(
      "Grid type",
      grid_type,
      "Type of grid to use. Options are 'grid_generator' and 'unstructured'.");
    add_parameter(
      "Number of initial refinements",
      n_refinements,
      "Number of global refinements to perform on the initial mesh.");
    add_parameter("Number of refinements cycles",
                  n_ref_cycles,
                  "Number of cycles to perform.");
    add_parameter("Use piston mesh", use_piston);
  }
  leave_subsection();

  enter_subsection("R3MG");
  {
    add_parameter("Partitioner type", partitioner_type);
    add_parameter("MG Starting level", mg_starting_level);
    add_parameter("Extraction level", extraction_level);
    add_parameter("Smoother steps", smoother_steps);
    add_parameter("Keep ratio constant", keep_ratio_constant);
  }
  leave_subsection();

  outer_solver_control.declare_parameters_call_back.connect([]() -> void {
    ParameterAcceptor::prm.set("Max steps", "100");
    ParameterAcceptor::prm.set("Tolerance", "1.e-9");
    ParameterAcceptor::prm.set("Reduction", "1.e-6");
    ParameterAcceptor::prm.set("Log history", "true");
    ParameterAcceptor::prm.set("Log result", "true");
  });
}



template <int dim>
class Poisson
{
private:
  void
  make_grid();
  void
  assemble_system();
  void
  setup_multigrid();
  void
  check_amg();
  void
  local_refinement();
  void
  test_agglo_mg_with_cells();


  Triangulation<dim>            tria;
  const ProblemParameters<dim> &parameters;
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
  Poisson(const ProblemParameters<dim> &);
  void
  run();

  std::string  grid_type;
  std::string  partitioner_type;
  std::string  solution_type;
  unsigned int extraction_level;
  std::string  output_info_filename;

  DoFHandler<dim>                   original_dof_handler;
  std::vector<SparseMatrix<double>> injection_matrices;
  std::vector<SparsityPattern>      injection_sparsity_patterns;
  ReductionControl                  solver_control;

  // Only this for cells agglomeration
  static constexpr unsigned int rtree_m_cells =
    dim == 2 ? 2 : 4; // 2D: 2, 3D: 4

  static constexpr unsigned int rtree_m_points = 2; // 4;
  // m = 4 for 3D, m = 2 for 2D  Q1 elements

  // TODO: fix depending on dim and k (es. m=4 for Q2 in 2D). See sandbox

  static constexpr unsigned int rtree_M_cells  = 2 * rtree_m_cells;
  static constexpr unsigned int rtree_M_points = 2 * rtree_m_points;
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
Poisson<dim>::Poisson(const ProblemParameters<dim> &problem_parameters)
  : parameters(problem_parameters)
  ,
#ifdef HEX
  mapping()
#else
  mapping(FE_SimplexP<dim>{1})
#endif
  , fe_q(parameters.fe_degree)
  , grid_type(parameters.grid_type)
  , partitioner_type(parameters.partitioner_type)
  , solution_type(parameters.solution_type)
  , extraction_level(parameters.extraction_level)
  , output_info_filename(parameters.output_directory + "/output_info_" +
                         sanitize_for_filename(parameters.grid_type) + "_p" +
                         std::to_string(parameters.fe_degree) + "_cp" +
                         std::to_string(parameters.coarse_fe_degree) + "_ref" +
                         std::to_string(parameters.n_refinements) + ".txt")
  , original_dof_handler(tria)
  , solver_control(parameters.outer_solver_control)
{
  bool is_valid_m = false;
  if constexpr (dim == 3)
    is_valid_m = rtree_m_cells >= 4 ? true : false;
  else if constexpr (dim == 2)
    is_valid_m = rtree_m_cells == 2 ? true : false;
  else
    DEAL_II_NOT_IMPLEMENTED();

  AssertThrow(
    is_valid_m,
    ExcMessage(
      "Invalid m for R-tree partitioning. Adjust parameter m accordingly to the dimension."));

  if (solution_type == "linear")
    analytical_solution = std::make_unique<SolutionLinear<dim>>();
  else if (solution_type == "quadratic")
    analytical_solution = std::make_unique<SolutionQuadratic<dim>>();
  else if (solution_type == "product")
    analytical_solution = std::make_unique<SolutionProduct<dim>>();
  else if (solution_type == "product_sine")
    analytical_solution = std::make_unique<SolutionProductSine<dim>>();

  rhs_function = std::make_unique<const RightHandSide<dim>>(solution_type);
  constraints.close();
}


template <int dim>
void
Poisson<dim>::make_grid()
{
  GridIn<dim> grid_in;
  if (grid_type == "unstructured")
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
          tria.refine_global(parameters.n_refinements);
        }
      else if constexpr (dim == 3)
        {
          grid_in.attach_triangulation(tria);
#ifdef HEX
          if (parameters.use_piston)
            {
              std::ifstream filename(
                "../../meshes/piston_3.inp"); // piston mesh
              grid_in.read_abaqus(filename);
              tria.refine_global(parameters.n_refinements);
              std::cout << "Minimal tria mesh size pre-scaling: "
                        << GridTools::minimal_cell_diameter(tria)
                        << " Maximal tria mesh size pre-scaling: "
                        << GridTools::maximal_cell_diameter(tria) << std::endl;
            }
          else
            {
              std::ifstream filename("../../meshes/realistic_lv.msh");
              grid_in.read_msh(filename);
              tria.refine_global(parameters.n_refinements);
              std::cout << "Minimal tria mesh size pre-scaling: "
                        << GridTools::minimal_cell_diameter(tria)
                        << " Maximal tria mesh size pre-scaling: "
                        << GridTools::maximal_cell_diameter(tria) << std::endl;
              GridTools::scale(1e-2, tria);
            }

          AssertThrow(tria.all_reference_cells_are_hyper_cube(),
                      ExcMessage("Mixed mesh. Bailing out"));
#else
          std::ifstream filename(
            "../../meshes/gray_level_image1.vtk"); // liver or brain domain
          grid_in.read_vtk(filename);

#endif
        }
    }
  else
    {
      // Grids generated through using GridGenerator
#ifdef HEX

      if constexpr (dim == 2)
        GridGenerator::hyper_cube(tria, 0., 1.);
      // GridGenerator::hyper_ball(tria, Point<dim>(), 1.);
      else if constexpr (dim == 3)
        GridGenerator::hyper_cube(tria, 0., 1.);
      // GridGenerator::eccentric_hyper_shell(tria,
      //                                      Point<dim>(1., 1., 1.),
      //                                      Point<dim>(0.7, 0.7, 0.7),
      //                                      0.2,
      //                                      1.,
      //                                      12 /*cells along circumference*/);
      tria.refine_global(parameters.n_refinements);
#else
      Triangulation<dim> tria_hex;
      GridGenerator::hyper_cube(tria_hex, 0., 1.);
      tria_hex.refine_global(parameters.n_refinements);
      GridGenerator::convert_hypercube_to_simplex_mesh(tria_hex, tria);
#endif
    }

  cached_tria = std::make_unique<GridTools::Cache<dim>>(tria, mapping);

  if (partitioner_type == "no_partition" || partitioner_type == "metis" ||
      partitioner_type == "rtree")
    {
    }
  else
    {
      Assert(false, ExcMessage("Wrong partitioning."));
    }
}



template <int dim>
void
Poisson<dim>::assemble_system()
{
  std::cout
    << "======================= Assembly of differential operator on the starting grid ==================="
    << std::endl;
  std::cout << "Size of tria: " << tria.n_active_cells() << std::endl;
  original_dof_handler.distribute_dofs(fe_q);

  std::cout << "Number of Boundary IDs: " << tria.get_boundary_ids().size()
            << std::endl;
  for (const auto &id : tria.get_boundary_ids())
    std::cout << " - Boundary ID: " << id << std::endl;

  constraints.clear();
  DoFTools::make_hanging_node_constraints(original_dof_handler, constraints);
  if (grid_type == "unstructured" && dim == 3)
    {
      if (parameters.use_piston)
        {
          VectorTools::interpolate_boundary_values(original_dof_handler,
                                                   types::boundary_id(0),
                                                   *analytical_solution,
                                                   constraints);
          VectorTools::interpolate_boundary_values(original_dof_handler,
                                                   types::boundary_id(1),
                                                   *analytical_solution,
                                                   constraints);
          VectorTools::interpolate_boundary_values(original_dof_handler,
                                                   types::boundary_id(2),
                                                   *analytical_solution,
                                                   constraints);
        }
      else
        {
          VectorTools::interpolate_boundary_values(original_dof_handler,
                                                   types::boundary_id(10),
                                                   *analytical_solution,
                                                   constraints);
          VectorTools::interpolate_boundary_values(original_dof_handler,
                                                   types::boundary_id(20),
                                                   *analytical_solution,
                                                   constraints);
          VectorTools::interpolate_boundary_values(original_dof_handler,
                                                   types::boundary_id(50),
                                                   *analytical_solution,
                                                   constraints);
        }
    }
  else
    {
      VectorTools::interpolate_boundary_values(original_dof_handler,
                                               types::boundary_id(0),
                                               *analytical_solution,
                                               constraints);
      std::cout << "Applied Dirichlet BCs on boundary ID 0" << std::endl;
    }

  constraints.close();

  dsp.reinit(original_dof_handler.n_dofs(), original_dof_handler.n_dofs());

  DoFTools::make_sparsity_pattern(original_dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ true);

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

  std::cout << "Started assembly" << std::endl;
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

      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
  std::cout << "Finished assembly" << std::endl;

  std::cout << "Original tria has " << original_dof_handler.n_dofs() << " DoFs."
            << std::endl;
  std::cout << "Built finest system matrix with dimensions "
            << system_matrix.m() << " x " << system_matrix.n() << std::endl;
  // std::string   filename = std::string("system_matrix.txt");
  // std::ofstream outfile(filename);
  // system_matrix.print_as_numpy_arrays(outfile);
  // outfile.close();

  {
    // Let's print the fine triangulation
    GridOut       grid_out;
    std::ofstream out("fine_tria.vtk");
    grid_out.write_vtk(tria, out);
  }
}



template <int dim>
void
Poisson<dim>::setup_multigrid()
{
  namespace bgi                                   = boost::geometry::index;
  static constexpr unsigned int min_elem_per_node = rtree_m_points;
  static constexpr unsigned int max_elem_per_node = rtree_M_points;
  static constexpr bool         use_points        = true;
  FE_DGQ<dim>                   fe_dg(parameters.coarse_fe_degree);
  double                        start, stop;

  start = MPI_Wtime();

  std::vector<Point<dim>> support_points_vector(original_dof_handler.n_dofs());

  DoFTools::map_dofs_to_support_points(mapping,
                                       original_dof_handler,
                                       support_points_vector);

  auto tree =
    pack_rtree_of_indices<bgi::rstar<max_elem_per_node, min_elem_per_node>>(
      support_points_vector);
  std::cout
    << "======================= Agglomeration with points multigrid testing ==================="
    << std::endl;

  std::cout << "Number of levels in the tree: " << n_levels(tree) << std::endl;

  unsigned int leaves_level      = n_levels(tree);
  bool         skip_leaves_level = true;

  if (skip_leaves_level)
    {
      leaves_level = n_levels(tree) - 1;
      std::cout << "Skipping leaves level is ON" << std::endl;
    }

  std::cout << "Total number allowed for MG: " << leaves_level << std::endl;

  std::vector<std::vector<BoundingBox<dim>>> all_level_boxes(leaves_level);

  // This cycle creates all the bounding boxes at each level
  for (unsigned int i = 0; i < leaves_level; ++i)
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
  triangulations.reserve(leaves_level);

  std::vector<std::unique_ptr<DoFHandler<dim>>> all_level_support_DoFHandlers;
  all_level_support_DoFHandlers.reserve(leaves_level);


  for (unsigned int i = 0; i < leaves_level; ++i)
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
  AssertThrow(all_level_support_DoFHandlers.size() == leaves_level,
              ExcMessage(
                "Inconsistent number of DoFHandlers for multigrid levels"));

  // Output the Bboxes trias at each level
  std::cout << "Outputting bounding box trias at each level" << std::endl;
  for (unsigned int level = 0; level < leaves_level; ++level)
    {
      GridOut       grid_out;
      std::ofstream out("point_bboxes_level_" + std::to_string(level) + ".vtk");
      grid_out.write_vtk(*triangulations[level], out);

      std::cout << "h_min at level " << level << " is "
                << GridTools::minimal_cell_diameter(*triangulations[level])
                << std::endl;
      std::cout << "h_max at level " << level << " is "
                << GridTools::maximal_cell_diameter(*triangulations[level])
                << std::endl;
    }

  std::cout << "h_min at level " << leaves_level + 1 << " is "
            << GridTools::minimal_cell_diameter(tria) << std::endl;
  std::cout << "h_max at level " << leaves_level + 1 << " is "
            << GridTools::maximal_cell_diameter(tria) << std::endl;

  injection_matrices.clear();
  injection_sparsity_patterns.clear();
  injection_matrices.resize(leaves_level);
  injection_sparsity_patterns.resize(leaves_level);

  for (unsigned int level = 0; level < leaves_level - 1; ++level)
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
      tree, leaves_level};
    const std::vector<std::vector<types::global_dof_index>> agglomerates =
      agglomerator_test.extract_agglomerates();

    std::vector<types::global_dof_index> dof_indices_agglo_tria(
      fe_dg.n_dofs_per_cell());

    DynamicSparsityPattern dsp_agglo_to_original_tria;
    dsp_agglo_to_original_tria.reinit(
      original_dof_handler.n_dofs(),
      all_level_support_DoFHandlers[leaves_level - 1]->n_dofs());

    unsigned int agglo_index = 0;
    for (const auto &cell : all_level_support_DoFHandlers[leaves_level - 1]
                              ->active_cell_iterators())
      {
        cell->get_dof_indices(dof_indices_agglo_tria);

        for (const types::global_dof_index dof_idx : agglomerates[agglo_index])
          dsp_agglo_to_original_tria.add_entries(dof_idx,
                                                 dof_indices_agglo_tria.begin(),
                                                 dof_indices_agglo_tria.end());

        ++agglo_index;
      }

    // Now build the matrix...
    injection_sparsity_patterns[leaves_level - 1].copy_from(
      dsp_agglo_to_original_tria);
    injection_matrices[leaves_level - 1].reinit(
      injection_sparsity_patterns[leaves_level - 1]);

    AffineConstraints<double> dummy_constraints;

    // reset agglo_index
    agglo_index = 0;
    for (const auto &cell : all_level_support_DoFHandlers[leaves_level - 1]
                              ->active_cell_iterators())
      {
        cell->get_dof_indices(dof_indices_agglo_tria);

        const BoundingBox<dim> &coarse_box =
          all_level_boxes[leaves_level - 1][cell->active_cell_index()];

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

        dummy_constraints.distribute_local_to_global(
          local_matrix2,
          fine_indices,           // tria original
          dof_indices_agglo_tria, // agglomerated tria
          injection_matrices[leaves_level - 1]);

        ++agglo_index; // advance to next agglomerate
      }

    std::cout << "Built transfer matrix agglo to original tria with dimensions "
              << injection_matrices[leaves_level - 1].m() << " x "
              << injection_matrices[leaves_level - 1].n() << std::endl;
  }


  std::cout << "Finished setting up multigrid transfer operators" << std::endl;

  // Output all transfer matrices for numpy
  // for (unsigned int level = 0; level < leaves_level; ++level)
  //   {
  //     std::string filename_tr =
  //       std::string("point_transfer_matrix_level_") +
  //       Utilities::int_to_string(level) + std::string("_to_") +
  //       Utilities::int_to_string(level + 1) + std::string(".txt");
  //     std::ofstream outfile_tr(filename_tr);
  //     injection_matrices[level].print_as_numpy_arrays(outfile_tr);
  //     outfile_tr.close();
  //   }

  // Let's add an output for paraview printing some basis functions. The output
  // should be on the original fine tria
  // std::cout << "Output some basis functions on the original fine tria"
  //           << std::endl;
  // {
  //   for (unsigned int level = 0; level < n_levels(tree); ++level)
  //     {
  //       for (unsigned int shape_fun_idx = 0; shape_fun_idx < 4;
  //       ++shape_fun_idx)
  //         {
  //           Vector<double> source_shape_fun(injection_matrices[level].n());
  //           source_shape_fun[shape_fun_idx] = 1.0;

  //           for (unsigned int inner_level = level; inner_level <
  //           n_levels(tree);
  //                ++inner_level)
  //             {
  //               Vector<double> target_shape_fun(
  //                 injection_matrices[inner_level].m());

  //               injection_matrices[inner_level].vmult(target_shape_fun,
  //                                                     source_shape_fun);

  //               source_shape_fun.reinit(injection_matrices[inner_level].m());
  //               source_shape_fun = target_shape_fun;
  //             }

  //           {
  //             DataOut<dim> data_out;
  //             data_out.attach_dof_handler(original_dof_handler);

  //             data_out.add_data_vector(
  //               source_shape_fun,
  //               "shape_function_level_" + Utilities::int_to_string(level) +
  //                 "_idx_" + Utilities::int_to_string(shape_fun_idx),
  //               DataOut<dim>::type_dof_data);

  //             data_out.build_patches(mapping);

  //             const std::string filename =
  //               "basis_function_level_" + Utilities::int_to_string(level) +
  //               "_idx_" + Utilities::int_to_string(shape_fun_idx) + ".vtu";
  //             std::ofstream output(filename);
  //             data_out.write_vtu(output);
  //           }
  //         }
  //     }
  // }

  if (parameters.mg_starting_level > leaves_level)
    throw std::runtime_error(
      "mg_starting_level is larger than available levels in the agglomeration tree");

  std::cout << "Support points vector size original tria: "
            << support_points_vector.size() << std::endl;

  std::cout << "----------------------------------------" << std::endl;
  std::cout << "Setting up multigrid from level "
            << parameters.mg_starting_level << " to level " << leaves_level + 1
            << std::endl;

  std::vector<TrilinosWrappers::SparseMatrix> trilinos_transfer_matrices(
    leaves_level - parameters.mg_starting_level + 1);

  // Copy everything to Trilinos matrices to use already existing stuff
  for (unsigned int level = 0;
       level < leaves_level - parameters.mg_starting_level + 1;
       ++level)
    {
      trilinos_transfer_matrices[level].reinit(
        injection_matrices[level + parameters.mg_starting_level - 1]);
    }

  AmgProjector<dim, TrilinosWrappers::SparseMatrix, double> amg_projector(
    trilinos_transfer_matrices); // Initialize projector
  std::cout << "Initialized AMG projector" << std::endl;

  MGLevelObject<std::unique_ptr<TrilinosWrappers::SparseMatrix>>
    multigrid_matrices(0, leaves_level - parameters.mg_starting_level + 1);

  multigrid_matrices[multigrid_matrices.max_level()] =
    std::make_unique<TrilinosWrappers::SparseMatrix>();

  // Set up finest level system matrix (copy the matrix content)
  multigrid_matrices[multigrid_matrices.max_level()]->reinit(system_matrix);
  std::cout << "Built finest operator" << std::endl;

  amg_projector.compute_level_matrices(multigrid_matrices);
  std::cout << "Projected using transfer_matrices:" << std::endl;

  std::cout << "Check dimensions of level operators" << std::endl;
  for (unsigned int level = 0; level <= multigrid_matrices.max_level(); ++level)
    std::cout << "Level " << level + 1 + parameters.mg_starting_level - 1
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
  smoother_data.resize(0, leaves_level + 1 - parameters.mg_starting_level + 1);

  std::cout << "Setting up smoothers" << std::endl;
  std::cout << "Setting up finest level smoother at level " << leaves_level + 1
            << std::endl;

  VectorType diag_inverse(system_matrix.m());
  for (unsigned int row = 0; row < system_matrix.m(); ++row)
    diag_inverse[row] = 1. / system_matrix.diag_element(row);
  diag_inverse.compress(VectorOperation::insert);

  std::vector<VectorType> diag_inverses(leaves_level + 1 -
                                        parameters.mg_starting_level + 1);
  diag_inverses[leaves_level + 1 - parameters.mg_starting_level] = diag_inverse;

  smoother_data[leaves_level - parameters.mg_starting_level + 1]
    .preconditioner = std::make_shared<DiagonalMatrix<VectorType>>(
    diag_inverses[leaves_level - parameters.mg_starting_level + 1]);


  for (unsigned int level = 0;
       level < leaves_level - parameters.mg_starting_level + 1;
       ++level)
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

      std::cout << "Level " << level + 1 + parameters.mg_starting_level - 1
                << " smoother set up " << std::endl;
    }

  std::cout << "Initialized smoothers data" << std::endl;

  for (unsigned int level = 0;
       level < leaves_level + 1 - parameters.mg_starting_level + 1;
       ++level)
    {
      if (level > 0)
        {
          smoother_data[level].smoothing_range = 20.; // 15.;
          smoother_data[level].degree          = 3; // parameters.smoother_steps
          smoother_data[level].eig_cg_n_iterations = 20;
        }
      else
        {
          smoother_data[0].smoothing_range = 1e-3;
          smoother_data[0].degree = 3; // numbers::invalid_unsigned_int;
          smoother_data[0].eig_cg_n_iterations = 20;
        }
    }

  mg_smoother.set_steps(parameters.smoother_steps);
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
    0, leaves_level - parameters.mg_starting_level + 1);
  for (unsigned int l = 0; l < leaves_level - parameters.mg_starting_level + 1;
       ++l)
    mg_level_transfers[l] = &trilinos_transfer_matrices[l];

  std::vector<DoFHandler<dim> *> dof_handlers(leaves_level + 1 -
                                              parameters.mg_starting_level + 1);
  for (unsigned int l = 0; l < dof_handlers.size() - 1; ++l)
    dof_handlers[l] =
      all_level_support_DoFHandlers[l + parameters.mg_starting_level - 1].get();
  dof_handlers[leaves_level - parameters.mg_starting_level + 1] =
    &original_dof_handler;

  unsigned int lev = parameters.mg_starting_level;
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

  stop = MPI_Wtime();

  VectorType dist_solution;
  VectorType dist_rhs;
  dist_solution.reinit(original_dof_handler.n_dofs());
  dist_rhs.reinit(original_dof_handler.n_dofs());
  for (unsigned int i = 0; i < system_rhs.size(); ++i)
    dist_rhs[i] = system_rhs[i];
  dist_rhs.compress(VectorOperation::insert);

  SolverCG<VectorType> cg(solver_control);


  std::ofstream file(output_info_filename, std::ios::app);
  if (file.is_open())
    {
      file << "\n[Point Agglomeration AMG]" << std::endl;
      write_stat_line(file,
                      "Global refinements",
                      std::to_string(parameters.n_refinements));
      write_stat_line(file,
                      "R-tree m parameter",
                      std::to_string(rtree_m_points));
      write_stat_line(file, "R-tree levels", std::to_string(n_levels(tree)));
      write_stat_line(file,
                      "MG level range",
                      std::to_string(parameters.mg_starting_level) + " -> " +
                        std::to_string(leaves_level));
      write_stat_line(file,
                      "Total MG levels",
                      std::to_string(leaves_level -
                                     parameters.mg_starting_level + 2));

      std::vector<double> H_h_vector;

      // extract H_level/H_level+1 for all levels from starting level to leaves
      // level and output the maximum
      for (unsigned int h_level = parameters.mg_starting_level - 1;
           h_level + 1 < leaves_level;
           ++h_level)
        {
          H_h_vector.push_back(
            GridTools::maximal_cell_diameter(*triangulations[h_level]) /
            GridTools::maximal_cell_diameter(*triangulations[h_level + 1]));
        }
      H_h_vector.push_back(
        GridTools::maximal_cell_diameter(*triangulations[leaves_level - 1]) /
        GridTools::minimal_cell_diameter(tria));

      for (double val : H_h_vector)
        std::cout << "H_level/H_level+1: " << val << " ";
      std::cout << std::endl;

      {
        std::ostringstream out;
        out << std::setprecision(6)
            << *std::max_element(H_h_vector.begin(), H_h_vector.end());
        write_stat_line(file, "Max H(l)/H(l+1)", out.str());
      }

      // file << "H max at starting level over h max at finest level: "
      //      << GridTools::maximal_cell_diameter(
      //           *triangulations[parameters.mg_starting_level - 1]) /
      //           GridTools::maximal_cell_diameter(tria)
      //      << std::endl;

      // double H_avg = (GridTools::minimal_cell_diameter(
      //                   *triangulations[parameters.mg_starting_level - 1]) +
      //                 GridTools::maximal_cell_diameter(
      //                   *triangulations[parameters.mg_starting_level - 1])) /
      //                2.0;
      // double h_avg = (GridTools::minimal_cell_diameter(tria) +
      //                 GridTools::maximal_cell_diameter(tria)) /
      //                2.0;

      // file << "H averaged at starting level over h averaged at finest level:
      // "
      //      << H_avg / h_avg << std::endl;

      {
        std::ostringstream out;
        out << std::fixed << std::setprecision(6) << stop - start;
        write_stat_line(file, "Setup time [s]", out.str());
      }
    }

  const std::string point_output_file = output_info_filename;
  cg.connect_condition_number_slot([point_output_file](double input) {
    std::ofstream file(point_output_file, std::ios::app);
    if (file.is_open())
      {
        std::ostringstream out;
        out << std::setprecision(6) << input;
        write_stat_line(file, "Condition number estimate", out.str());
      }
  });

  std::cout << "Start solver" << std::endl;
  start = MPI_Wtime();
  cg.solve(system_matrix, dist_solution, dist_rhs, preconditioner);
  stop = MPI_Wtime();
  std::cout << "Point Agglo AMG elapsed time: " << stop - start << "[s]"
            << std::endl;

  std::cout << "Initial value: " << solver_control.initial_value() << std::endl;
  std::cout << "Converged in " << solver_control.last_step()
            << " iterations with value " << solver_control.last_value()
            << std::endl;

  if (file.is_open())
    {
      write_stat_line(file,
                      "IterationsPointAggloAMG",
                      std::to_string(solver_control.last_step()));
      {
        std::ostringstream out;
        out << std::scientific << std::setprecision(6)
            << solver_control.last_value();
        write_stat_line(file, "Final residual", out.str());
      }
      {
        std::ostringstream out;
        out << std::fixed << std::setprecision(6) << stop - start;
        write_stat_line(file, "Solve time [s]", out.str());
      }
      file.close();
    }

  // Copy back the solution inside the class solution vector
  for (unsigned int i = 0; i < solution.size(); ++i)
    solution[i] = dist_solution[i];

  constraints.distribute(solution);

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

  // Check that solution is close to the analytical solution
  {
    Vector<double> difference_per_cell(tria.n_active_cells());

    VectorTools::integrate_difference(original_dof_handler,
                                      solution,
                                      *analytical_solution,
                                      difference_per_cell,
                                      QGauss<dim>(fe_q.degree + 1),
                                      VectorTools::L2_norm);

    const double L2_error =
      difference_per_cell.l2_norm(); // global L2 norm of the error

    std::cout << "L2 error compared to analytical solution: " << L2_error
              << std::endl;


    {
      std::ofstream      file(output_info_filename, std::ios::app);
      std::ostringstream out;
      out << std::scientific << std::setprecision(6) << L2_error;
      write_stat_line(file, "L2 error", out.str());
    }
  }
}



template <int dim>
void
Poisson<dim>::check_amg()
{
  using VectorType = LinearAlgebra::distributed::Vector<double>;
  double start, stop;

  std::cout << "Checking standard AMG from Trilinos" << std::endl;

  // start = MPI_Wtime();

  TrilinosWrappers::PreconditionAMG                 prec_amg;
  TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;

  amg_data.aggregation_threshold = 1e-2; // AMG aggregation threshold
  amg_data.smoother_type         = "Chebyshev";
  amg_data.smoother_sweeps       = parameters.smoother_steps;
  amg_data.output_details        = true;

  if (fe_q.get_degree() > 1)
    amg_data.higher_order_elements = true;

  TrilinosWrappers::SparseMatrix system_matrix_trilinos;
  system_matrix_trilinos.reinit(system_matrix);

  prec_amg.initialize(system_matrix_trilinos, amg_data);

  VectorType dist_solution;
  VectorType dist_rhs;
  dist_solution.reinit(original_dof_handler.n_dofs());
  dist_rhs.reinit(original_dof_handler.n_dofs());
  for (unsigned int i = 0; i < system_rhs.size(); ++i)
    dist_rhs[i] = system_rhs[i];
  dist_rhs.compress(VectorOperation::insert);

  SolverCG<VectorType> cg_check(solver_control);

  start = MPI_Wtime();
  cg_check.solve(system_matrix_trilinos, dist_solution, dist_rhs, prec_amg);
  stop = MPI_Wtime();
  // Copy back the solution inside the class solution vector
  for (unsigned int i = 0; i < solution.size(); ++i)
    solution[i] = dist_solution[i];

  constraints.distribute(solution);

  //  stop = MPI_Wtime();

  std::cout << "Initial value: " << solver_control.initial_value() << std::endl;
  std::cout << "Converged (CG+AMG) in " << solver_control.last_step()
            << " iterations with value " << solver_control.last_value()
            << std::endl;

  std::ofstream file(output_info_filename, std::ios::app);
  if (file.is_open())
    {
      file << "\n[Trilinos AMG]" << std::endl;
      write_stat_line(file,
                      "IterationsAMG",
                      std::to_string(solver_control.last_step()));
      {
        std::ostringstream out;
        out << std::scientific << std::setprecision(6)
            << solver_control.last_value();
        write_stat_line(file, "Final residual", out.str());
      }
      {
        std::ostringstream out;
        out << std::fixed << std::setprecision(6) << stop - start;
        write_stat_line(file, "Solve time [s]", out.str());
      }

      // Check that solution is close to the analytical solution

      Vector<double> difference_per_cell(tria.n_active_cells());

      VectorTools::integrate_difference(original_dof_handler,
                                        solution,
                                        *analytical_solution,
                                        difference_per_cell,
                                        QGauss<dim>(fe_q.degree + 1),
                                        VectorTools::L2_norm);

      const double L2_error =
        difference_per_cell.l2_norm(); // global L2 norm of the error

      std::cout << "L2 error compared to analytical solution: " << L2_error
                << std::endl;

      std::ostringstream out;
      out << std::scientific << std::setprecision(6) << L2_error;
      write_stat_line(file, "L2 error", out.str());
    }
  file << std::string(80, '-') << std::endl;
  file.close();
}



template <int dim>
void
Poisson<dim>::local_refinement()
{
  Vector<float> estimated_error_per_cell(tria.n_active_cells());

  KellyErrorEstimator<dim>::estimate(original_dof_handler,
                                     QGauss<dim - 1>(fe_q.degree + 1),
                                     {},
                                     solution,
                                     estimated_error_per_cell);

  GridRefinement::refine_and_coarsen_fixed_number(tria,
                                                  estimated_error_per_cell,
                                                  0.3,
                                                  0.03);

  tria.execute_coarsening_and_refinement();
}


// This should be ok but check in with Marco
template <int dim>
void
create_bounding_box_from_agglo_cells(
  std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>
                                &vec_agglomerates,
  std::vector<BoundingBox<dim>> &agglomerate_boxes)
{
  MappingQ1<dim> mapping;
  agglomerate_boxes.reserve(vec_agglomerates.size());

  for (const auto &agglo : vec_agglomerates)
    {
      bool       init = false;
      Point<dim> p_min, p_max;

      for (const auto &cell : agglo)
        {
          const auto &bb = mapping.get_bounding_box(cell);
          const auto &bp = bb.get_boundary_points(); // {min,max}

          if (!init)
            {
              p_min = bp.first;
              p_max = bp.second;
              init  = true;
            }
          else
            {
              for (unsigned int d = 0; d < dim; ++d)
                {
                  p_min[d] = std::min(p_min[d], bp.first[d]);
                  p_max[d] = std::max(p_max[d], bp.second[d]);
                }
            }
        }
      agglomerate_boxes.emplace_back(std::make_pair(p_min, p_max));
    }
}



template <int dim>
void
Poisson<dim>::test_agglo_mg_with_cells()
{
  std::cout
    << "======================= Agglomeration with cells multigrid testing ==================="
    << std::endl;
  namespace bgi = boost::geometry::index;

  static constexpr unsigned int min_elem_per_node = rtree_m_cells;
  static constexpr unsigned int max_elem_per_node = rtree_M_cells;
  FE_DGQ<dim>                   fe_dg(parameters.coarse_fe_degree);
  double                        start, stop;

  start = MPI_Wtime();

  std::vector<std::pair<BoundingBox<dim>,
                        typename Triangulation<dim>::active_cell_iterator>>
    boxes(tria.n_active_cells());

  unsigned int i = 0;
  for (const auto &cell : tria.active_cell_iterators())
    boxes[i++] = std::make_pair(mapping.get_bounding_box(cell), cell);

  auto tree =
    pack_rtree<bgi::rstar<max_elem_per_node, min_elem_per_node>>(boxes);
  std::cout << "Number of levels in the tree: " << n_levels(tree) << std::endl;

  unsigned int leaves_level      = n_levels(tree);
  bool         skip_leaves_level = true;

  if (skip_leaves_level)
    {
      leaves_level = n_levels(tree) - 1;
      std::cout << "Skipping leaves level is ON" << std::endl;
    }

  std::cout << "Total number allowed for MG: " << leaves_level << std::endl;

  std::vector<std::vector<BoundingBox<dim>>> all_level_boxes(leaves_level);

  // This cycle creates all the bounding boxes at each agglo level
  for (unsigned int i = 0; i < leaves_level; ++i)
    {
      CellsAgglomerator<dim, decltype(tree)> agglomerator{tree, i + 1};

      std::vector<
        std::vector<typename Triangulation<dim>::active_cell_iterator>>
        agglomerates = agglomerator.extract_agglomerates();
      all_level_boxes[i].reserve(agglomerates.size());

      create_bounding_box_from_agglo_cells(agglomerates, all_level_boxes[i]);
    }

  std::cout << "Finished creating bounding boxes for multigrid levels"
            << std::endl;

  std::vector<std::unique_ptr<Triangulation<dim>>> triangulations;
  triangulations.reserve(leaves_level);

  std::vector<std::unique_ptr<DoFHandler<dim>>> all_level_support_DoFHandlers;
  all_level_support_DoFHandlers.reserve(leaves_level);

  for (unsigned int i = 0; i < leaves_level; ++i)
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
  AssertThrow(all_level_support_DoFHandlers.size() == leaves_level,
              ExcMessage(
                "Inconsistent number of DoFHandlers for multigrid levels"));

  // Output the Bboxes trias at each level
  std::cout << "Outputting bounding box trias at each level" << std::endl;
  for (unsigned int level = 0; level < leaves_level; ++level)
    {
      GridOut       grid_out;
      std::ofstream out("bboxes_level_" + std::to_string(level + 1) + ".vtk");
      grid_out.write_vtk(*triangulations[level], out);

      std::cout << "h_min at level " << level + 1 << " is "
                << GridTools::minimal_cell_diameter(*triangulations[level])
                << std::endl;
      std::cout << "h_max at level " << level + 1 << " is "
                << GridTools::maximal_cell_diameter(*triangulations[level])
                << std::endl;
    }

  std::cout << "h_min at level " << leaves_level + 1 << " is "
            << GridTools::minimal_cell_diameter(tria) << std::endl;
  std::cout << "h_max at level " << leaves_level + 1 << " is "
            << GridTools::maximal_cell_diameter(tria) << std::endl;

  injection_matrices.clear();
  injection_sparsity_patterns.clear();
  injection_matrices.resize(leaves_level);
  injection_sparsity_patterns.resize(leaves_level);

  for (unsigned int level = 0; level < leaves_level - 1; ++level)
    {
      CellsAgglomerator<dim, decltype(tree)> agglomerator{tree, level + 1};
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

  // Build leaves transfer matrix
  {
    CellsAgglomerator<dim, decltype(tree)> leaves_agglomerator{tree,
                                                               leaves_level};

    std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>
      leaves_vec_agglomerates = leaves_agglomerator.extract_agglomerates();

    DynamicSparsityPattern dsp_leaves_to_original;
    dsp_leaves_to_original.reinit(
      original_dof_handler.n_dofs(),
      all_level_support_DoFHandlers[leaves_level - 1]->n_dofs());

    std::vector<types::global_dof_index> dof_indices_agglo_leaves_tria(
      fe_dg.n_dofs_per_cell());
    std::vector<types::global_dof_index> dof_indices_original_tria(
      fe_q.n_dofs_per_cell());

    IndexSet assigned_dofs(original_dof_handler.n_dofs());

    if (assigned_dofs.n_elements() != 0)
      throw std::runtime_error(
        "Assigned dofs index set should be empty at this point");

    for (const auto &cell : all_level_support_DoFHandlers[leaves_level - 1]
                              ->active_cell_iterators())
      {
        cell->get_dof_indices(dof_indices_agglo_leaves_tria);

        for (const auto &child_cell :
             leaves_vec_agglomerates[cell->active_cell_index()])
          {
            // unsigned int cell_idx = child_cell->active_cell_index();

            const auto child_cell_dh =
              child_cell->as_dof_handler_iterator(original_dof_handler);

            child_cell_dh->get_dof_indices(dof_indices_original_tria);

            for (const auto &fine_dof_idx : dof_indices_original_tria)
              {
                if (assigned_dofs.is_element(fine_dof_idx))
                  continue;
                else
                  {
                    dsp_leaves_to_original.add_entries(
                      fine_dof_idx,
                      dof_indices_agglo_leaves_tria.begin(),
                      dof_indices_agglo_leaves_tria.end());

                    assigned_dofs.add_index(fine_dof_idx);
                  }
              }
          }
      }

    if (assigned_dofs.n_elements() != original_dof_handler.n_dofs())
      {
        throw std::runtime_error(
          "Not all DoFs have been assigned during sparsity generation in the transfer from leaves agglo to original tria");
      }

    injection_sparsity_patterns[leaves_level - 1].copy_from(
      dsp_leaves_to_original);
    injection_matrices[leaves_level - 1].reinit(
      injection_sparsity_patterns[leaves_level - 1]);

    // Reset assigned_dofs to start filling the matrix with values
    assigned_dofs.clear();
    if (assigned_dofs.n_elements() != 0)
      throw std::runtime_error(
        "Assigned dofs index set should be empty at this point");

    AffineConstraints<double> dummy_constraints;

    std::vector<Point<dim>> unit_support_points =
      fe_q.get_unit_support_points();

    for (const auto &cell : all_level_support_DoFHandlers[leaves_level - 1]
                              ->active_cell_iterators())
      {
        cell->get_dof_indices(dof_indices_agglo_leaves_tria);

        const BoundingBox<dim> &coarse_box =
          all_level_boxes[leaves_level - 1][cell->active_cell_index()];

        std::vector<Point<dim>> local_support_points;
        // at most we can have number of support points per cell * number of
        // cells in the agglo
        local_support_points.reserve(
          unit_support_points.size() *
          leaves_vec_agglomerates[cell->active_cell_index()].size());

        std::vector<types::global_dof_index> actual_dof_indices_original_tria;

        actual_dof_indices_original_tria.reserve(
          unit_support_points.size() *
          leaves_vec_agglomerates[cell->active_cell_index()].size());

        for (const auto &child_cell :
             leaves_vec_agglomerates[cell->active_cell_index()])
          {
            const auto child_cell_dh =
              child_cell->as_dof_handler_iterator(original_dof_handler);

            child_cell_dh->get_dof_indices(dof_indices_original_tria);

            unsigned int find_dof_counter = 0;
            for (const auto &fine_dof_idx : dof_indices_original_tria)
              {
                if (!assigned_dofs.is_element(fine_dof_idx))
                  {
                    assigned_dofs.add_index(fine_dof_idx);
                    actual_dof_indices_original_tria.push_back(fine_dof_idx);
                    Point<dim> real_pt = mapping.transform_unit_to_real_cell(
                      child_cell, unit_support_points[find_dof_counter]);
                    local_support_points.push_back(real_pt);
                  }
                find_dof_counter++;
              }
          }

        FullMatrix<double> local_matrix2(local_support_points.size(),
                                         fe_dg.n_dofs_per_cell());
        local_matrix2 = 0.;

        for (unsigned int i = 0; i < local_support_points.size(); ++i)
          {
            const Point<dim> p =
              coarse_box.real_to_unit(local_support_points[i]);
            for (unsigned int j = 0; j < dof_indices_agglo_leaves_tria.size();
                 ++j)
              {
                local_matrix2(i, j) = fe_dg.shape_value(j, p);
              }
          }

        dummy_constraints.distribute_local_to_global(
          local_matrix2,
          actual_dof_indices_original_tria,
          dof_indices_agglo_leaves_tria,
          injection_matrices[leaves_level - 1]);
      }

    if (assigned_dofs.n_elements() != original_dof_handler.n_dofs())
      {
        throw std::runtime_error(
          "Not all DoFs have been assigned during matrix filling in the transfer from leaves agglo to original tria");
      }

    std::cout << "Built transfer matrix agglo to original tria with dimensions "
              << injection_matrices[leaves_level - 1].m() << " x "
              << injection_matrices[leaves_level - 1].n() << std::endl;
  }

  std::cout << "Finished setting up multigrid transfer operators" << std::endl;
  // Output all transfer matrices for numpy
  // for (unsigned int level = 0; level < leaves_level; ++level)
  //   {
  //     std::string filename_tr =
  //       std::string("transfer_matrix_level_") +
  //       Utilities::int_to_string(level) + std::string("_to_") +
  //       Utilities::int_to_string(level + 1) + std::string(".txt");
  //     std::ofstream outfile_tr(filename_tr);
  //     injection_matrices[level].print_as_numpy_arrays(outfile_tr);
  //     outfile_tr.close();
  //   }

  // mg_starting_level
  if (parameters.mg_starting_level > leaves_level)
    throw std::runtime_error(
      "mg_starting_level is larger than available levels in the agglomeration tree");

  std::cout << "----------------------------------------" << std::endl;

  std::cout << "Setting up multigrid from level "
            << parameters.mg_starting_level << " to level " << leaves_level + 1
            << std::endl;
  std::vector<TrilinosWrappers::SparseMatrix> trilinos_transfer_matrices(
    leaves_level - parameters.mg_starting_level + 1);
  // Copy everything to Trilinos matrices to use already existing stuff
  for (unsigned int level = 0;
       level < leaves_level - parameters.mg_starting_level + 1;
       ++level)
    {
      trilinos_transfer_matrices[level].reinit(
        injection_matrices[level + parameters.mg_starting_level - 1]);
    }

  AmgProjector<dim, TrilinosWrappers::SparseMatrix, double> amg_projector(
    trilinos_transfer_matrices); // Initialize projector
  std::cout << "Initialized AMG projector" << std::endl;

  MGLevelObject<std::unique_ptr<TrilinosWrappers::SparseMatrix>>
    multigrid_matrices(0, leaves_level - parameters.mg_starting_level + 1);

  multigrid_matrices[multigrid_matrices.max_level()] =
    std::make_unique<TrilinosWrappers::SparseMatrix>();

  // Set up finest level system matrix (copy the matrix content)
  multigrid_matrices[multigrid_matrices.max_level()]->reinit(system_matrix);
  std::cout << "Built finest operator" << std::endl;

  amg_projector.compute_level_matrices(multigrid_matrices);
  std::cout << "Projected using transfer_matrices:" << std::endl;

  std::cout << "Check dimensions of level operators" << std::endl;
  for (unsigned int level = 0; level <= multigrid_matrices.max_level(); ++level)
    std::cout << "Level " << level + 1 + parameters.mg_starting_level - 1
              << " operator size: " << multigrid_matrices[level]->m() << " x "
              << multigrid_matrices[level]->n() << std::endl;

  using LevelMatrixType = TrilinosWrappers::SparseMatrix;
  using VectorType      = LinearAlgebra::distributed::Vector<double>;
  mg::Matrix<VectorType> mg_matrix(multigrid_matrices);

  using SmootherType = PreconditionChebyshev<LevelMatrixType, VectorType>;
  mg::SmootherRelaxation<SmootherType, VectorType>     mg_smoother;
  MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
  smoother_data.resize(0, leaves_level + 1 - parameters.mg_starting_level + 1);

  std::cout << "Setting up smoothers" << std::endl;
  std::cout << "Setting up finest level smoother at level " << leaves_level + 1
            << std::endl;

  VectorType diag_inverse(system_matrix.m());
  for (unsigned int row = 0; row < system_matrix.m(); ++row)
    diag_inverse[row] = 1. / system_matrix.diag_element(row);
  diag_inverse.compress(VectorOperation::insert);

  std::vector<VectorType> diag_inverses(leaves_level + 1 -
                                        parameters.mg_starting_level + 1);
  diag_inverses[leaves_level - parameters.mg_starting_level + 1] = diag_inverse;

  smoother_data[leaves_level - parameters.mg_starting_level + 1]
    .preconditioner = std::make_shared<DiagonalMatrix<VectorType>>(
    diag_inverses[leaves_level - parameters.mg_starting_level + 1]);

  for (unsigned int level = 0;
       level < leaves_level - parameters.mg_starting_level + 1;
       ++level)
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

      std::cout << "Level " << level + 1 + parameters.mg_starting_level - 1
                << " smoother set up " << std::endl;
    }

  std::cout << "Initialized smoothers data" << std::endl;

  for (unsigned int level = 0;
       level < leaves_level + 1 - parameters.mg_starting_level + 1;
       ++level)
    {
      if (level > 0)
        {
          smoother_data[level].smoothing_range     = 20.; // 15.;
          smoother_data[level].degree              = 3;   // 5;
          smoother_data[level].eig_cg_n_iterations = 20;
        }
      else
        {
          smoother_data[0].smoothing_range = 1e-3;
          smoother_data[0].degree = 3; // numbers::invalid_unsigned_int;
          smoother_data[0].eig_cg_n_iterations = 20;
        }
    }

  mg_smoother.set_steps(parameters.smoother_steps);
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
    0, leaves_level - parameters.mg_starting_level + 1);
  for (unsigned int l = 0; l < leaves_level - parameters.mg_starting_level + 1;
       ++l)
    mg_level_transfers[l] = &trilinos_transfer_matrices[l];

  std::vector<DoFHandler<dim> *> dof_handlers(leaves_level + 1 -
                                              parameters.mg_starting_level + 1);
  // Align MG level indexing with the chosen mg_starting_level:
  // mg-level 0 corresponds to agglomeration level
  // `parameters.mg_starting_level`.
  for (unsigned int l = 0; l < dof_handlers.size() - 1; ++l)
    dof_handlers[l] =
      all_level_support_DoFHandlers[l + parameters.mg_starting_level - 1].get();
  // Finest level corresponds to the original DoFHandler
  dof_handlers[leaves_level - parameters.mg_starting_level + 1] =
    &original_dof_handler;

  unsigned int lev = parameters.mg_starting_level;
  for (const auto &dh : dof_handlers)
    {
      std::cout << "Number of DoFs in level " << lev << ": " << dh->n_dofs()
                << std::endl;
      ++lev;
    }

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

  stop = MPI_Wtime();

  VectorType dist_solution;
  VectorType dist_rhs;
  dist_solution.reinit(original_dof_handler.n_dofs());
  dist_rhs.reinit(original_dof_handler.n_dofs());
  for (unsigned int i = 0; i < system_rhs.size(); ++i)
    dist_rhs[i] = system_rhs[i];
  dist_rhs.compress(VectorOperation::insert);
  SolverCG<VectorType> cg(solver_control);

  std::ofstream file(output_info_filename, std::ios::app);
  if (file.is_open())
    {
      file << "\n[Cell Agglomeration AMG]" << std::endl;
      write_stat_line(file,
                      "Global refinements",
                      std::to_string(parameters.n_refinements));
      write_stat_line(file,
                      "R-tree m parameter",
                      std::to_string(rtree_m_cells));
      write_stat_line(file, "R-tree levels", std::to_string(n_levels(tree)));
      write_stat_line(file,
                      "MG level range",
                      std::to_string(parameters.mg_starting_level) + " -> " +
                        std::to_string(leaves_level));
      write_stat_line(file,
                      "Total MG levels",
                      std::to_string(leaves_level -
                                     parameters.mg_starting_level + 2));

      std::vector<double> H_h_vector;

      // extract H_level/H_level+1 for all levels from starting level to leaves
      // level and output the maximum
      for (unsigned int h_level = parameters.mg_starting_level - 1;
           h_level + 1 < leaves_level;
           ++h_level)
        {
          H_h_vector.push_back(
            GridTools::maximal_cell_diameter(*triangulations[h_level]) /
            GridTools::maximal_cell_diameter(*triangulations[h_level + 1]));
        }
      H_h_vector.push_back(
        GridTools::maximal_cell_diameter(*triangulations[leaves_level - 1]) /
        GridTools::minimal_cell_diameter(tria));

      for (double val : H_h_vector)
        std::cout << "H_level/H_level+1: " << val << " ";
      std::cout << std::endl;

      {
        std::ostringstream out;
        out << std::setprecision(6)
            << *std::max_element(H_h_vector.begin(), H_h_vector.end());
        write_stat_line(file, "Max H(l)/H(l+1)", out.str());
      }

      // file << "H max at starting level over h max at finest level: "
      //      << GridTools::maximal_cell_diameter(
      //           *triangulations[parameters.mg_starting_level - 1]) /
      //           GridTools::maximal_cell_diameter(tria)
      //      << std::endl;

      // double H_avg = (GridTools::minimal_cell_diameter(
      //                   *triangulations[parameters.mg_starting_level - 1]) +
      //                 GridTools::maximal_cell_diameter(
      //                   *triangulations[parameters.mg_starting_level - 1])) /
      //                2.0;
      // double h_avg = (GridTools::minimal_cell_diameter(tria) +
      //                 GridTools::maximal_cell_diameter(tria)) /
      //                2.0;

      // file << "H averaged at starting level over h averaged at finest level:
      // "
      //      << H_avg / h_avg << std::endl;

      {
        std::ostringstream out;
        out << std::fixed << std::setprecision(6) << stop - start;
        write_stat_line(file, "Setup time [s]", out.str());
      }
    }

  const std::string cell_output_file = output_info_filename;
  cg.connect_condition_number_slot([cell_output_file](double input) {
    std::ofstream file(cell_output_file, std::ios::app);
    if (file.is_open())
      {
        std::ostringstream out;
        out << std::setprecision(6) << input;
        write_stat_line(file, "Condition number estimate", out.str());
      }
  });

  std::cout << "Start solver" << std::endl;
  start = MPI_Wtime();
  cg.solve(system_matrix, dist_solution, dist_rhs, preconditioner);
  stop = MPI_Wtime();
  std::cout << "Agglo AMG cells elapsed time: " << stop - start << "[s]"
            << std::endl;

  std::cout << "Initial value: " << solver_control.initial_value() << std::endl;
  std::cout << "Converged in " << solver_control.last_step()
            << " iterations with value " << solver_control.last_value()
            << std::endl;

  if (file.is_open())
    {
      write_stat_line(file,
                      "IterationsCellsAggloAMG",
                      std::to_string(solver_control.last_step()));
      {
        std::ostringstream out;
        out << std::scientific << std::setprecision(6)
            << solver_control.last_value();
        write_stat_line(file, "Final residual", out.str());
      }
      {
        std::ostringstream out;
        out << std::fixed << std::setprecision(6) << stop - start;
        write_stat_line(file, "Solve time [s]", out.str());
      }
      file.close();
    }


  // Copy back the solution inside the class solution vector
  for (unsigned int i = 0; i < solution.size(); ++i)
    solution[i] = dist_solution[i];

  constraints.distribute(solution);

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

  // if (original_dof_handler.n_dofs() < 3e6)
  //   output_results();

  // Check that solution is close to the analytical solution
  {
    Vector<double> difference_per_cell(tria.n_active_cells());

    VectorTools::integrate_difference(original_dof_handler,
                                      solution,
                                      *analytical_solution,
                                      difference_per_cell,
                                      QGauss<dim>(fe_q.degree + 1),
                                      VectorTools::L2_norm);

    const double L2_error =
      difference_per_cell.l2_norm(); // global L2 norm of the error

    std::cout << "L2 error compared to analytical solution: " << L2_error
              << std::endl;
    {
      std::ofstream      file(output_info_filename, std::ios::app);
      std::ostringstream out;
      out << std::scientific << std::setprecision(6) << L2_error;
      write_stat_line(file, "L2 error", out.str());
    }
  }
}



template <int dim>
void
Poisson<dim>::run()
{
  std::filesystem::create_directories(parameters.output_directory);
  {
    std::ofstream file(output_info_filename, std::ios::app);
    if (file.is_open())
      {
        file << "\n" << std::string(80, '=') << std::endl;
        file << "Run timestamp: " << current_time_string() << std::endl;
        file << "Dimension    : " << dim << std::endl;
        file << "Grid type    : " << parameters.grid_type << std::endl;
        file << "FE degree    : " << parameters.fe_degree << std::endl;
        file << "Coarse degree: " << parameters.coarse_fe_degree << std::endl;
        file << "Refinements  : " << parameters.n_refinements << std::endl;
        file << "Partitioner  : " << parameters.partitioner_type << std::endl;
        file << "Output file  : " << output_info_filename << std::endl;
        file << std::string(80, '=') << std::endl;
      }
  }

  make_grid();
  auto start = std::chrono::high_resolution_clock::now();
  assemble_system();
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
    std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  std::cout << "Time taken by assemble_system(): " << duration.count() / 1e6
            << " seconds" << std::endl;

  setup_multigrid();
  test_agglo_mg_with_cells();

  check_amg();

  // std::ofstream file("output_info.txt", std::ios::app);
  // if (file.is_open())
  //   {
  //     file << "==========================================" << std::endl;
  //     file << "Test after local refinement: " << std::endl;
  //     file.close();
  //   }

  // std::cout << "==========================================" << std::endl;
  // std::cout << "Test after local refinement: " << std::endl;
  // local_refinement();
  // assemble_system();
  // setup_multigrid();
  // test_agglo_mg_with_cells();
  // check_amg();
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  deallog.depth_console(10);

  static constexpr unsigned int dim = 2;
  ProblemParameters<dim>        parameters;
  std::string                   parameter_file;
  if (argc > 1)
    parameter_file = argv[1];
  else
    parameter_file = "parameters_MG.prm";
  ParameterAcceptor::initialize(parameter_file, "used_parameters.prm");

  // std::vector<double> starting_level_vector = {6};
  // parameters.mg_starting_level = starting_level_vector[s_level_counter];
  // s_level_counter++;
  // unsigned int s_level_counter = 0;


  for (unsigned int cycle = 0; cycle < parameters.n_ref_cycles; ++cycle)
    {
      Poisson<dim> poisson_problem{parameters};
      poisson_problem.run();
      parameters.n_refinements++;
      if (parameters.keep_ratio_constant)
        ++parameters.mg_starting_level;
    }

  std::cout << std::endl;
  return 0;
}
